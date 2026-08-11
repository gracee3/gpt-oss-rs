use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use gpt_oss_cpu_kernels::{
    KernelPath, Kernels, Mxfp4Block, Q8Block, ResidualQ8Block, QUANT_BLOCK_SIZE,
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

criterion_group!(benches, benchmark_bf16_matvec, benchmark_mxfp4_q8);
criterion_main!(benches);
