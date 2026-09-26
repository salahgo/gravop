# F5 — solver smoothness and integration-method pilot

2026-09-10. This is a new diagnostic experiment; F4's failed convergence result,
cap and thresholds remain unchanged. Model weights, production and manuscript
remain frozen. F4 observations may guide numerical diagnostics but cannot serve
as fresh confirmation after choosing a new method.

## First stage: targeted temporal refinement

Use F4's final mesh and observation likelihoods. Select the three cells with
largest normalized embedded error among minimum-size cells, the three largest
among refinable cells, and the largest contributing cell for the worst-error
observation at each of the three times. Deduplicate cells and their nine nodes.
Preserve the selected identities before generating any refined profiles.

Repeat each selected direct trajectory with eps_t=1e-4 and 3e-5, respectively,
and snapshot spacings (dlnrho, dlnt)=(0.01,0.025) and (0.005,0.0125). Compare
against the cached eps_t=3e-4 reference. Preserve all statuses and support
disagreements. Numerical failures block downstream integration. Report maximum
and RMS log-density changes on common support, terminal-time changes, and the
worst profile changes relative to observational noise (0.0434 dex). A temporal
screen requires zero support disagreements and maximum profile change <=0.00434
dex (one tenth of observational noise) between successive levels. This is a
diagnostic screen, not proof of solver accuracy, spatial convergence or posterior
convergence. Near-boundary support disagreements must be investigated, not
silently excluded. Do not change the existing data-generation settings globally.

## Second stage: evidence-led integration pilot

Choose and record the concrete pilot only after the solver diagnostics. Candidate
approaches include nested higher-order positive rules using existing nodes, or
per-observation integration. The [SciPy cubature documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html)
describes Gauss–Kronrod and Genz–Malik subdivision and explicitly does not guarantee
convergence. The [SciPy integration tutorial](https://docs.scipy.org/doc/scipy/tutorial/integrate.html)
warns that narrow important regions may be missed. Any method must retain the
full original prior, explicitly account for survival support, and check posterior
CDFs as well as moments and evidence. Do not use a neural approximation as the
direct-solver reference. If temporal instability is material, resolve that before
interpreting a new integration result as improved calibration.

Analytic tests and a bounded solver pilot precede broad fresh confirmation.
Record all failed gates and distinguish diagnostic reuse from independent
confirmation. No production replacement is authorized by this experiment alone.
