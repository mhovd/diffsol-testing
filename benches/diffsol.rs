use pmcore::prelude::*;
fn ode() -> ODE {
    equation::ODE::new(
        |x, p, t, dx, rateiv, cov| {
            fetch_params!(p, cls, k30, k40, qs, vps, vs, fm1, fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let ke = cl / v;
            let _vm1 = vfrac1 * v;
            let _vm2 = vfrac2 * v;
            let k12 = q / v;
            let k21 = q / vp;

            //</tem>
            dx[0] = rateiv[0] - ke * x[0] * (1.0 - fm1 - fm2) - (fm1 + fm2) * x[0] - k12 * x[0]
                + k21 * x[1];
            dx[1] = k12 * x[0] - k21 * x[1];
            dx[2] = fm1 * x[0] - k30 * x[2];
            dx[3] = fm2 * x[0] - k40 * x[3];
        },
        |_p| {
            lag! {}
        },
        |_p| {
            fa! {}
        },
        |_p, _t, _cov, _x| {},
        |x, p, t, cov, y| {
            fetch_params!(p, cls, _k30, _k40, qs, vps, vs, _fm1, _fm2, theta1, theta2);
            fetch_cov!(cov, t, wt, pkvisit);

            let vfrac1 = 0.068202;
            let vfrac2 = 0.022569;
            let cl = cls * ((pkvisit - 1.0) * theta1).exp() * (wt / 70.0).powf(0.75);
            let q = qs * (wt / 70.0).powf(0.75);
            let v = vs * ((pkvisit - 1.0) * theta2).exp() * (wt / 70.0);
            let vp = vps * (wt / 70.0);
            let _ke = cl / v;
            let vm1 = vfrac1 * v;
            let vm2 = vfrac2 * v;
            let _k12 = q / v;
            let _k21 = q / vp;

            y[0] = x[0] / v;
            y[1] = x[2] / vm1;
            y[2] = x[3] / vm2;
        },
        (4, 3),
    )
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
