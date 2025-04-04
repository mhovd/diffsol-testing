use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
type M = nalgebra::DMatrix<f64>;
type V = nalgebra::DVector<f64>;
type S = f64;

// lotka_volterra
//  F: Fn(&M::V, &M::V, M::T, &mut M::V),
fn rhs() -> impl Fn(&V, &V, S, &mut V) {
    |x: &V, p: &V, _t: S, y: &mut V| {
        if p.len() != 4 {
            dbg!(&p);
        }
        let alpha: f64 = p[0];
        let beta = p[1];
        let delta = p[2];
        let gamma = p[3];

        y[0] = alpha * x[0] - beta * x[0] * x[1];
        y[1] = delta * x[0] * x[1] - gamma * x[1];
    }
}
fn jac() -> impl Fn(&V, &V, S, &V, &mut V) {
    |x: &V, p: &V, _t: S, v: &V, y: &mut V| {
        let alpha = p[0];
        let beta = p[1];
        let delta = p[2];
        let gamma = p[3];

        y[0] = (alpha - beta * x[1]) * v[0] - beta * x[0] * v[1];
        y[1] = delta * x[1] * v[0] + (delta * x[0] - gamma) * v[1];
    }
}

fn run_diffsol_053(n: usize) {
    use diffsol::OdeSolverMethod;
    use nalgebra::DVector;

    use diffsol::OdeBuilder;
    type LS = diffsol::NalgebraLU<f64>;
    let problem = OdeBuilder::<M>::new()
        .p(vec![1.0, 0.1, 0.1, 1.0])
        .rhs_implicit(rhs(), jac())
        .init(|_p, _t| DVector::from_vec(vec![10.0, 5.0]))
        .build()
        .unwrap();

    let mut solver = problem.bdf::<LS>().unwrap();
    let (ys, _ts) = solver.solve(24.0 * n as f64).unwrap();
    black_box(ys);
}

fn run_diffsol_030(n: usize) {
    use diffsol_old::{Bdf, OdeBuilder, OdeSolverMethod, OdeSolverState};
    use nalgebra::DVector;

    let problem = OdeBuilder::new()
        .p(vec![1.0, 0.1, 0.1, 1.0])
        .build_ode::<M, _, _, _>(rhs(), jac(), |_p, _t| DVector::from_vec(vec![10.0, 5.0]))
        .unwrap();
    let mut solver = Bdf::default();
    let state = OdeSolverState::new(&problem, &solver).unwrap();
    let (ys, _ts) = solver.solve(&problem, state, 24.0 * n as f64).unwrap();
    black_box(ys);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Comparison");
    for &n in &[2048, 4096] {
        group.bench_with_input(BenchmarkId::new("Diffsol 0.5.3", n), &n, |b, &n| {
            b.iter(|| run_diffsol_053(black_box(n)))
        });
        group.bench_with_input(BenchmarkId::new("Diffsol 0.3.0", n), &n, |b, &n| {
            b.iter(|| run_diffsol_030(black_box(n)))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
