/// Kernel fusion intermediate representation.
///
/// Represents a DAG of GPU operations and identifies chains that can be fused
/// into single CUDA kernels, keeping data in registers/shared memory instead
/// of round-tripping through HBM.
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

pub type NodeId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dtype {
    F16,
    F32,
}

impl Dtype {
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// Primitive operations that can appear in the fusion graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FusionOp {
    /// RMS normalization with epsilon for numerical stability.
    RMSNorm { eps: f32 },
    /// Matrix-vector multiply (M=1 decode path).
    Gemv,
    /// Add a bias vector.
    BiasAdd,
    /// SiLU activation (x * sigmoid(x)).
    SiLU,
    /// Elementwise multiply.
    ElemMul,
    /// Elementwise add.
    ElemAdd,
    /// Rotary positional embedding.
    RoPE,
    /// Softmax (full reduction -- acts as a fusion barrier).
    Softmax,
    /// Identity / passthrough copy.
    Copy,
}

impl FusionOp {
    /// True if this op is purely elementwise (no reductions, no large state).
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            FusionOp::SiLU
                | FusionOp::ElemMul
                | FusionOp::ElemAdd
                | FusionOp::BiasAdd
                | FusionOp::Copy
        )
    }

    /// True if this op is a full-tensor reduction that forces a sync point.
    pub fn is_barrier(&self) -> bool {
        matches!(self, FusionOp::Softmax)
    }
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: FusionOp,
    /// Indices of upstream nodes whose outputs feed into this node.
    pub input_ids: Vec<NodeId>,
    /// Shape of the output tensor (row-major).
    pub output_shape: Vec<usize>,
    pub dtype: Dtype,
}

// ---------------------------------------------------------------------------
// FusionGraph -- the DAG
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionGraph {
    pub nodes: Vec<Node>,
}

impl FusionGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a node and return its id.
    pub fn add_node(
        &mut self,
        op: FusionOp,
        input_ids: Vec<NodeId>,
        output_shape: Vec<usize>,
        dtype: Dtype,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            id,
            op,
            input_ids,
            output_shape,
            dtype,
        });
        id
    }

    /// Return the immediate consumers of a given node.
    pub fn consumers(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|n| n.input_ids.contains(&id))
            .map(|n| n.id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Fusion analysis
    // -----------------------------------------------------------------------

    /// Returns true if `producer -> consumer` is a fusible edge according to
    /// the fusion rules:
    ///
    /// 1. Elementwise ops can always fuse with adjacent ops.
    /// 2. RMSNorm can fuse into a following Gemv (norm+projection).
    /// 3. SiLU followed by ElemMul can fuse into a following Gemv.
    /// 4. Gemv CANNOT fuse with another Gemv (register pressure).
    /// 5. Softmax is a barrier -- nothing fuses across it.
    fn can_fuse(&self, producer: NodeId, consumer: NodeId) -> bool {
        let p = &self.nodes[producer];
        let c = &self.nodes[consumer];

        // Barrier ops block fusion in both directions.
        if p.op.is_barrier() || c.op.is_barrier() {
            return false;
        }

        // Gemv -> Gemv is never fusible (too much register pressure).
        if matches!(p.op, FusionOp::Gemv) && matches!(c.op, FusionOp::Gemv) {
            return false;
        }

        // Elementwise ops fuse with anything non-barrier.
        if p.op.is_elementwise() || c.op.is_elementwise() {
            return true;
        }

        // RMSNorm -> Gemv (norm + projection).
        if matches!(p.op, FusionOp::RMSNorm { .. }) && matches!(c.op, FusionOp::Gemv) {
            return true;
        }

        // RoPE is fusible with elementwise (already covered) and with Gemv.
        if matches!(p.op, FusionOp::RoPE) || matches!(c.op, FusionOp::RoPE) {
            return true;
        }

        false
    }

    /// Walk the graph and find maximal groups of fusible operations.
    ///
    /// Each returned `FusedKernel` contains a topologically-ordered list of
    /// node ids that should be compiled into a single CUDA kernel.
    ///
    /// Algorithm: process nodes in topological order. For each node, try to
    /// merge it into a producer's group if:
    ///   1. The edge is fusible (`can_fuse`).
    ///   2. Adding this node wouldn't violate constraints (e.g. two Gemvs).
    /// If no producer group accepts it, start a new group.
    pub fn find_fusible_chains(&self) -> Vec<FusedKernel> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        // group_of[node_id] = index into `groups`
        let mut group_of: Vec<usize> = vec![0; n];
        // Each group: (node_ids, gemv_count)
        let mut groups: Vec<(Vec<NodeId>, usize)> = Vec::new();

        for id in 0..n {
            let node = &self.nodes[id];
            let is_gemv = matches!(node.op, FusionOp::Gemv);

            // Try to join a producer's group.
            let mut best_group: Option<usize> = None;
            for &inp in &node.input_ids {
                if !self.can_fuse(inp, id) {
                    continue;
                }
                let gid = group_of[inp];
                let (_, gemv_count) = &groups[gid];
                // Gemv cap: at most 1 Gemv per fused kernel.
                if is_gemv && *gemv_count >= 1 {
                    continue;
                }
                // Prefer joining the first viable producer group.
                if best_group.is_none() {
                    best_group = Some(gid);
                }
            }

            match best_group {
                Some(gid) => {
                    groups[gid].0.push(id);
                    if is_gemv {
                        groups[gid].1 += 1;
                    }
                    group_of[id] = gid;
                }
                None => {
                    group_of[id] = groups.len();
                    groups.push((vec![id], if is_gemv { 1 } else { 0 }));
                }
            }
        }

        groups
            .into_iter()
            .map(|(node_ids, _)| {
                let ops: Vec<FusionOp> =
                    node_ids.iter().map(|&id| self.nodes[id].op.clone()).collect();
                let output_node = &self.nodes[*node_ids.last().unwrap()];
                FusedKernel {
                    node_ids,
                    ops,
                    output_shape: output_node.output_shape.clone(),
                    dtype: output_node.dtype,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FusedKernel -- result of fusion analysis
// ---------------------------------------------------------------------------

/// A chain of operations to be compiled into a single CUDA kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedKernel {
    /// Topologically-ordered node ids from the original graph.
    pub node_ids: Vec<NodeId>,
    /// The corresponding ops, same order as `node_ids`.
    pub ops: Vec<FusionOp>,
    /// Output shape of the final op in the chain.
    pub output_shape: Vec<usize>,
    pub dtype: Dtype,
}

impl FusedKernel {
    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    /// True if every op in the kernel is elementwise (cheap register-only kernel).
    pub fn is_pure_elementwise(&self) -> bool {
        self.ops.iter().all(|op| op.is_elementwise())
    }

    /// Estimated register pressure category: low / medium / high.
    /// Downstream codegen uses this to decide tile sizes and occupancy targets.
    pub fn register_pressure(&self) -> RegisterPressure {
        let has_gemv = self.ops.iter().any(|op| matches!(op, FusionOp::Gemv));
        let has_norm = self
            .ops
            .iter()
            .any(|op| matches!(op, FusionOp::RMSNorm { .. }));

        if has_gemv {
            RegisterPressure::High
        } else if has_norm {
            RegisterPressure::Medium
        } else {
            RegisterPressure::Low
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegisterPressure {
    Low,
    Medium,
    High,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(d: usize) -> Vec<usize> {
        vec![1, d]
    }

    /// Gate-up MLP pattern: RMSNorm -> (Gemv_gate, Gemv_up) -> SiLU -> ElemMul -> Gemv_down
    #[test]
    fn test_mlp_fusion() {
        let mut g = FusionGraph::new();
        let norm = g.add_node(FusionOp::RMSNorm { eps: 1e-5 }, vec![], shape(4096), Dtype::F16);
        let gate = g.add_node(FusionOp::Gemv, vec![norm], shape(11008), Dtype::F16);
        let up = g.add_node(FusionOp::Gemv, vec![norm], shape(11008), Dtype::F16);
        let silu = g.add_node(FusionOp::SiLU, vec![gate], shape(11008), Dtype::F16);
        let mul = g.add_node(FusionOp::ElemMul, vec![silu, up], shape(11008), Dtype::F16);
        let _down = g.add_node(FusionOp::Gemv, vec![mul], shape(4096), Dtype::F16);

        let chains = g.find_fusible_chains();

        // Verify no chain contains two Gemv ops.
        for chain in &chains {
            let gemv_count = chain.ops.iter().filter(|op| matches!(op, FusionOp::Gemv)).count();
            assert!(gemv_count <= 1, "chain has {gemv_count} Gemv ops: {chain:?}");
        }

        // Norm should fuse with at least one Gemv.
        let norm_chain = chains.iter().find(|c| c.node_ids.contains(&norm)).unwrap();
        assert!(
            norm_chain.node_ids.contains(&gate) || norm_chain.node_ids.contains(&up),
            "RMSNorm did not fuse with a Gemv"
        );

        // SiLU and ElemMul should be in the same chain.
        let silu_chain = chains.iter().find(|c| c.node_ids.contains(&silu)).unwrap();
        assert!(silu_chain.node_ids.contains(&mul));
    }

    /// Softmax acts as a fusion barrier.
    #[test]
    fn test_softmax_barrier() {
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::ElemAdd, vec![], shape(128), Dtype::F32);
        let s = g.add_node(FusionOp::Softmax, vec![a], shape(128), Dtype::F32);
        let b = g.add_node(FusionOp::ElemMul, vec![s], shape(128), Dtype::F32);

        let chains = g.find_fusible_chains();

        // a, s, b must each be in separate chains.
        let chain_of = |id: NodeId| chains.iter().find(|c| c.node_ids.contains(&id)).unwrap();
        assert!(!chain_of(a).node_ids.contains(&s));
        assert!(!chain_of(s).node_ids.contains(&b));
        assert_eq!(chain_of(s).len(), 1);
    }

    #[test]
    fn test_pure_elementwise_chain() {
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::BiasAdd, vec![], shape(4096), Dtype::F16);
        let b = g.add_node(FusionOp::SiLU, vec![a], shape(4096), Dtype::F16);
        let _c = g.add_node(FusionOp::ElemAdd, vec![b], shape(4096), Dtype::F16);

        let chains = g.find_fusible_chains();
        assert_eq!(chains.len(), 1);
        assert!(chains[0].is_pure_elementwise());
        assert_eq!(chains[0].register_pressure(), RegisterPressure::Low);
    }

    #[test]
    fn test_register_pressure() {
        let mut g = FusionGraph::new();
        let norm = g.add_node(FusionOp::RMSNorm { eps: 1e-6 }, vec![], shape(4096), Dtype::F16);
        let _gemv = g.add_node(FusionOp::Gemv, vec![norm], shape(4096), Dtype::F16);

        let chains = g.find_fusible_chains();
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].register_pressure(), RegisterPressure::High);
    }

    #[test]
    fn test_empty_graph() {
        let g = FusionGraph::new();
        assert!(g.find_fusible_chains().is_empty());
    }
}
