use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gpt_oss_cpu_kernels::{
    e8m0_scale, mxfp4_adjacent_to_split, KernelPath, Kernels, Mxfp4Block, Mxfp4MatrixView,
    Mxfp4WeightLayout, Q8ActivationView, Q8Block, ResidualQ8ActivationView, ResidualQ8Block,
    QUANT_BLOCK_SIZE,
};
use half::bf16;

fn benchmark_bf16_matvec(criterion: &mut Criterion) {
    let rows = 256;
    let cols = 2048;
    let weights = (0..rows * cols)
        .map(|index| bf16::from_f32((index % 31) as f32 / 31.0 - 0.5))
        .collect::<Vec<_>>();
    let input = (0..cols)
        .map(|index| bf16::from_f32((index % 17) as f32 / 17.0 - 0.5))
        .collect::<Vec<_>>();
    let mut output = vec![0.0_f32; rows];
    let mut group = criterion.benchmark_group("bf16_matvec");
    group.throughput(Throughput::Elements((rows * cols) as u64));
    for path in [
        KernelPath::Auto,
        KernelPath::Scalar,
        KernelPath::Avx2,
        KernelPath::Avx512Vnni,
    ] {
        let Ok(kernels) = Kernels::new(path) else {
            continue;
        };
        group.bench_function(path.as_str(), |bencher| {
            bencher.iter(|| {
                kernels
                    .bf16_matvec(
                        black_box(&weights),
                        rows,
                        cols,
                        black_box(&input),
                        black_box(&mut output),
                    )
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn benchmark_mxfp4_q8(criterion: &mut Criterion) {
    let blocks = 64;
    let weights = vec![
        Mxfp4Block {
            scale: 127,
            packed: [0x52; QUANT_BLOCK_SIZE / 2],
        };
        blocks
    ];
    let activations = vec![
        Q8Block {
            scale: 0.01,
            values: [17; QUANT_BLOCK_SIZE],
        };
        blocks
    ];
    let residual_activations = vec![
        ResidualQ8Block {
            primary: activations[0].clone(),
            residual: Q8Block {
                scale: 0.0001,
                values: [-3; QUANT_BLOCK_SIZE],
            },
        };
        blocks
    ];
    let mut group = criterion.benchmark_group("mxfp4_q8_dot");
    group.throughput(Throughput::Elements((blocks * QUANT_BLOCK_SIZE) as u64));
    for path in [
        KernelPath::Auto,
        KernelPath::Scalar,
        KernelPath::Avx2,
        KernelPath::Avx512Vnni,
    ] {
        let Ok(kernels) = Kernels::new(path) else {
            continue;
        };
        group.bench_function(path.as_str(), |bencher| {
            bencher.iter(|| {
                black_box(kernels.mxfp4_q8_dot(black_box(&weights), black_box(&activations)))
            })
        });
        group.bench_function(format!("{}-residual", path.as_str()), |bencher| {
            bencher.iter(|| {
                black_box(
                    kernels.mxfp4_residual_q8_dot(
                        black_box(&weights),
                        black_box(&residual_activations),
                    ),
                )
            })
        });
    }
    group.finish();
}

fn projection_weights(rows: usize, blocks: usize) -> Vec<Mxfp4Block> {
    (0..rows * blocks)
        .map(|index| Mxfp4Block {
            scale: 118 + (index % 18) as u8,
            packed: std::array::from_fn(|byte| {
                let low = (index + byte * 2) as u8 & 0x0f;
                let high = (index * 3 + byte * 2 + 1) as u8 & 0x0f;
                low | (high << 4)
            }),
        })
        .collect()
}

fn pack_canonical(weights: &[Mxfp4Block]) -> Vec<u8> {
    let mut data = Vec::with_capacity(weights.len() * 17);
    for weight in weights {
        data.push(weight.scale);
        data.extend_from_slice(&weight.packed);
    }
    data
}

fn pack_x8(weights: &[Mxfp4Block], rows: usize, blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(weights.len() * 17);
    for group in 0..rows / 8 {
        for block in 0..blocks {
            for lane in 0..8 {
                data.push(weights[(group * 8 + lane) * blocks + block].scale);
            }
            let split = std::array::from_fn::<_, 8, _>(|lane| {
                mxfp4_adjacent_to_split(weights[(group * 8 + lane) * blocks + block].packed)
            });
            for chunk in 0..2 {
                for row in &split {
                    data.extend_from_slice(&row[chunk * 8..chunk * 8 + 8]);
                }
            }
        }
    }
    for row in rows / 8 * 8..rows {
        for block in 0..blocks {
            let weight = &weights[row * blocks + block];
            data.push(weight.scale);
            data.extend_from_slice(&weight.packed);
        }
    }
    data
}

fn legacy_q8_projection(
    kernels: Kernels,
    weights: &[Mxfp4Block],
    rows: usize,
    blocks: usize,
    activations: &[Q8Block],
    output: &mut [f32],
) {
    for (row, destination) in output.iter_mut().enumerate().take(rows) {
        let mut total = 0.0_f32;
        for block in 0..blocks {
            let weight = &weights[row * blocks + block];
            let integer = kernels.mxfp4_q8_block_dot_i32(weight, &activations[block]);
            total += integer as f32 * 0.5 * e8m0_scale(weight.scale) * activations[block].scale;
        }
        *destination = total;
    }
}

fn legacy_residual_projection(
    kernels: Kernels,
    weights: &[Mxfp4Block],
    rows: usize,
    blocks: usize,
    activations: &[ResidualQ8Block],
    output: &mut [f32],
) {
    for (row, destination) in output.iter_mut().enumerate().take(rows) {
        let mut total = 0.0_f32;
        for block in 0..blocks {
            let weight = &weights[row * blocks + block];
            let [primary, residual] =
                kernels.mxfp4_residual_q8_block_dot_i32(weight, &activations[block]);
            let weight_scale = 0.5 * e8m0_scale(weight.scale);
            total += primary as f32 * weight_scale * activations[block].primary.scale;
            total += residual as f32 * weight_scale * activations[block].residual.scale;
        }
        *destination = total;
    }
}

fn x8_q8_projection(
    kernels: Kernels,
    weights: Mxfp4MatrixView<'_>,
    activations: &[Q8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    for (tile, destination) in output.chunks_mut(8).enumerate() {
        kernels
            .mxfp4_q8_gemv_tile(
                weights,
                tile * 8,
                Q8ActivationView::new(activations),
                bias,
                destination,
            )
            .unwrap();
    }
}

fn x8_residual_projection(
    kernels: Kernels,
    weights: Mxfp4MatrixView<'_>,
    activations: &[ResidualQ8Block],
    bias: &[f32],
    output: &mut [f32],
) {
    for (tile, destination) in output.chunks_mut(8).enumerate() {
        kernels
            .mxfp4_residual_q8_gemv_tile(
                weights,
                tile * 8,
                ResidualQ8ActivationView::new(activations),
                bias,
                destination,
            )
            .unwrap();
    }
}

fn benchmark_projection_shape(criterion: &mut Criterion, shape: &str, rows: usize, cols: usize) {
    let blocks = cols / QUANT_BLOCK_SIZE;
    let weights = projection_weights(rows, blocks);
    let canonical = pack_canonical(&weights);
    let x8_packed = pack_x8(&weights, rows, blocks);
    let x8_view = Mxfp4MatrixView::new(
        &x8_packed,
        rows,
        blocks,
        Mxfp4WeightLayout::InterleavedSplitX8V2,
    )
    .unwrap();
    let q8 = (0..blocks)
        .map(|block| Q8Block {
            scale: 0.001 + block as f32 * 0.000001,
            values: std::array::from_fn(|lane| {
                (((block * 13 + lane * 7) % 255) as i16 - 127) as i8
            }),
        })
        .collect::<Vec<_>>();
    let residual = q8
        .iter()
        .cloned()
        .map(|primary| ResidualQ8Block {
            primary,
            residual: Q8Block {
                scale: 0.00001,
                values: std::array::from_fn(|lane| lane as i8 - 16),
            },
        })
        .collect::<Vec<_>>();
    let bias = vec![0.0; rows];
    let mut output = vec![0.0; rows];

    let mut packing = criterion.benchmark_group(format!("mxfp4_pack/{shape}"));
    packing.throughput(Throughput::Bytes((rows * blocks * 17) as u64));
    packing.bench_function("canonical", |bencher| {
        bencher.iter(|| black_box(pack_canonical(black_box(&weights))))
    });
    packing.bench_function("interleaved-split-x8", |bencher| {
        bencher.iter(|| black_box(pack_x8(black_box(&weights), rows, blocks)))
    });
    packing.finish();

    let scalar = Kernels::new(KernelPath::Scalar).unwrap();
    let avx2 = Kernels::new(KernelPath::Avx2).ok();
    let auto = Kernels::new(KernelPath::Auto).unwrap();
    let avx512 = Kernels::new(KernelPath::Avx512Vnni).ok();

    for (activation_name, residual_mode) in [("q8", false), ("residual-q8", true)] {
        let mut group = criterion.benchmark_group(format!("mxfp4_gemv/{shape}/{activation_name}"));
        group.throughput(Throughput::Bytes(canonical.len() as u64));
        group.bench_function("scalar-canonical", |bencher| {
            bencher.iter(|| {
                if residual_mode {
                    legacy_residual_projection(
                        scalar,
                        black_box(&weights),
                        rows,
                        blocks,
                        black_box(&residual),
                        black_box(&mut output),
                    )
                } else {
                    legacy_q8_projection(
                        scalar,
                        black_box(&weights),
                        rows,
                        blocks,
                        black_box(&q8),
                        black_box(&mut output),
                    )
                }
            })
        });
        if let Some(avx2) = avx2 {
            group.bench_function("legacy-avx2-canonical", |bencher| {
                bencher.iter(|| {
                    if residual_mode {
                        legacy_residual_projection(
                            avx2,
                            black_box(&weights),
                            rows,
                            blocks,
                            black_box(&residual),
                            black_box(&mut output),
                        )
                    } else {
                        legacy_q8_projection(
                            avx2,
                            black_box(&weights),
                            rows,
                            blocks,
                            black_box(&q8),
                            black_box(&mut output),
                        )
                    }
                })
            });
            group.bench_function("avx2-x8", |bencher| {
                bencher.iter(|| {
                    if residual_mode {
                        x8_residual_projection(
                            avx2,
                            x8_view,
                            black_box(&residual),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    } else {
                        x8_q8_projection(
                            avx2,
                            x8_view,
                            black_box(&q8),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    }
                })
            });
        }
        if auto.dispatch_plan().mxfp4_weight_layout() == Mxfp4WeightLayout::InterleavedSplitX8V2 {
            group.bench_function("auto", |bencher| {
                bencher.iter(|| {
                    if residual_mode {
                        x8_residual_projection(
                            auto,
                            x8_view,
                            black_box(&residual),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    } else {
                        x8_q8_projection(
                            auto,
                            x8_view,
                            black_box(&q8),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    }
                })
            });
        }
        if let Some(avx512) = avx512 {
            group.bench_function("avx512-vnni-canonical", |bencher| {
                bencher.iter(|| {
                    if residual_mode {
                        legacy_residual_projection(
                            avx512,
                            black_box(&weights),
                            rows,
                            blocks,
                            black_box(&residual),
                            black_box(&mut output),
                        )
                    } else {
                        legacy_q8_projection(
                            avx512,
                            black_box(&weights),
                            rows,
                            blocks,
                            black_box(&q8),
                            black_box(&mut output),
                        )
                    }
                })
            });
        }
        group.finish();

        let mut operations =
            criterion.benchmark_group(format!("mxfp4_gemv_ops/{shape}/{activation_name}"));
        let operations_per_weight = if residual_mode { 4 } else { 2 };
        operations.throughput(Throughput::Elements(
            (rows * cols * operations_per_weight) as u64,
        ));
        if let Some(avx2) = avx2 {
            operations.bench_function("avx2-x8", |bencher| {
                bencher.iter(|| {
                    if residual_mode {
                        x8_residual_projection(
                            avx2,
                            x8_view,
                            black_box(&residual),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    } else {
                        x8_q8_projection(
                            avx2,
                            x8_view,
                            black_box(&q8),
                            black_box(&bias),
                            black_box(&mut output),
                        )
                    }
                })
            });
        }
        operations.finish();
    }
}

fn benchmark_mxfp4_gemv(criterion: &mut Criterion) {
    benchmark_projection_shape(criterion, "gate-up-5760x2880", 5760, 2880);
    benchmark_projection_shape(criterion, "down-2880x2880", 2880, 2880);
}

criterion_group!(
    benches,
    benchmark_bf16_matvec,
    benchmark_mxfp4_q8,
    benchmark_mxfp4_gemv
);
criterion_main!(benches);
