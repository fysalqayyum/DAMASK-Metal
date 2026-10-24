# Compare two DAMASK grid results (same geometry/load): per output increment, volume-averaged P and
# the point-wise differences of P, F and the plastic state, relative to the reference run.
# usage: compare_runs.py ref.hdf5 other.hdf5 [phase]
import sys, h5py, numpy as np
ref, oth = h5py.File(sys.argv[1], 'r'), h5py.File(sys.argv[2], 'r')
incs = sorted([k for k in ref if k.startswith('increment') and k in oth], key=lambda s: int(s.split('_')[1]))
ph = sys.argv[3] if len(sys.argv) > 3 else list(ref[incs[-1]]['phase'].keys())[0]
print(f"{'increment':>12} {'<P11> ref MPa':>14} {'rel d<P>':>9} {'max|dP|/|P|':>11} {'rms|dP|/|P|':>11} {'max|dF|':>9}  state (max abs diff / max |ref|)")
for inc in incs[1:]:
    a, b = ref[inc]['phase'][ph]['mechanical'], oth[inc]['phase'][ph]['mechanical']
    Pa, Pb = a['P'][()], b['P'][()]
    nP = np.linalg.norm(Pa.reshape(-1, 9), axis=1)
    dP = np.linalg.norm((Pb - Pa).reshape(-1, 9), axis=1)
    mPa, mPb = Pa.mean(axis=0), Pb.mean(axis=0)
    s = ' '.join(f"{k} {np.abs(b[k][()] - a[k][()]).max() / max(np.abs(a[k][()]).max(), 1e-300):.1e}"
                 for k in a if k not in ('P', 'F', 'F_p', 'L_p', 'O') and k in b)
    print(f"{inc:>12} {mPa[0,0]/1e6:14.3f} {np.linalg.norm(mPb-mPa)/np.linalg.norm(mPa):9.1e} {(dP/nP.mean()).max():11.1e} "
          f"{np.sqrt((dP**2).mean())/nP.mean():11.1e} {np.abs(b['F'][()]-a['F'][()]).max():9.1e}  {s}")
