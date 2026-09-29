"""
Reusable event models and helpers for calcium/iGluSnFR response fitting.

Provides model callables and their default parameter specs to avoid redundancy
across demos and notebooks.

Available models:
- 'double_exp' (default): classic double exponential (constrained)
- 'cooperative': cooperative binding (Hill-like rise, exp decay)
- 'two_step_binding': biophysically accurate iGluSnFR model (binding + conformational change)
- 'single_exp': single exponential decay (constrained)
- 'alpha': alpha function (constrained)
- 'gamma': gamma function (constrained)
- 'bilinear': bilinear rise + exp decay (constrained)
- 'binding_kinetics': binding kinetics model with on/off rates + clearance (constrained)
- 'two_component': two-component model with shared rise time (constrained)
- 'desensitization': model with desensitization term (constrained)
- 'coop_plus_linear': cooperative binding + linear component (constrained)
- 'diffusion_clearance': diffusion rise + bi-exponential clearance (constrained)
- 'double_cooperative': sum of two cooperative binding components (constrained)
- 'hetero_coop': heterogeneous cooperative binding (constrained)
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, List
import numpy as np

# Global fit limits that can be tweaked by calling code. These bounds
# are applied across models for common parameters like t_onset and tau
# values.  Three tau categories:
#   - tau_primary: fast components (rise, fast decay): 0.5-10ms
#   - tau_secondary: slow components (slow decay): 10-300ms
#   - tau_tertiary: super-slow (accumulation, for tri-exp): 25-150ms
# NOTE: tau_secondary remains wide (10-300ms) for bi-exponential iglusnfr.
# For tri-exponential, tau_secondary is narrower (10-25ms) and tau_tertiary
# handles the super-slow component.
FIT_LIMITS: Dict[str, Tuple[float, float]] = {
    't_onset': (-2.0, 10.0),          # Peak location: -2 to +10 ms
    'tau': (0.001, 0.300),           # Generic tau up to 300ms
    'tau_primary': (0.0003, 0.010),  # Fast: 0.3-10ms (rise, fast decay) - allow very fast
    'tau_secondary': (0.008, 0.300), # Slow: 8-300ms (for bi-exponential or intermediate)
    'tau_tertiary': (0.020, 0.200),  # Super-slow: 20-200ms (for tri-exponential) - wider range
}


def set_fit_limits(**kwargs) -> None:
    """Update global fit limit tuples.

    Example::

        set_fit_limits(t_onset=(0, 8), tau_secondary=(0.020, 0.500))
    """
    for key, val in kwargs.items():
        if key in FIT_LIMITS and isinstance(val, (tuple, list)) and len(val) == 2:
            FIT_LIMITS[key] = (float(val[0]), float(val[1]))


def _apply_global_bounds(spec: Dict) -> Dict:
    """Override bounds for common parameters using FIT_LIMITS."""
    lb, ub = spec['bounds']
    lb = list(lb)
    ub = list(ub)
    for i, p in enumerate(spec['params']):
        if p == 't_onset':
            lb[i], ub[i] = FIT_LIMITS['t_onset']
        elif 'tau' in p:
            # Tri-exponential: fast < slow < superslow
            if 'superslow' in p or 'super_slow' in p:
                bounds = FIT_LIMITS.get('tau_tertiary')
            elif any(s in p for s in ('fast', 'rise', 'bind')) or p.endswith('1'):
                bounds = FIT_LIMITS.get('tau_primary')
            elif any(s in p for s in ('slow', 'conform', 'dissoc')) or p.endswith('2'):
                bounds = FIT_LIMITS.get('tau_secondary')
            elif 'decay' in p:
                # For generic 'tau_decay' without fast/slow qualifier, use secondary
                bounds = FIT_LIMITS.get('tau_secondary')
            else:
                bounds = FIT_LIMITS.get('tau')
            if bounds:
                lb[i], ub[i] = bounds
    spec['bounds'] = (lb, ub)
    return spec

def model_double_exp_constrained(t, amp, tau_rise, tau_decay, t_onset):
    """Classic double exponential: (exp(-t/tau_decay) - exp(-t/tau_rise)).

    Parameters in seconds; t in milliseconds.
    Normalizes by value at theoretical peak to keep 'amp' interpretable.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tr = max(tau_rise, 1e-6)
        td = max(tau_decay, 1e-6)
        if td <= tr:
            # Fall back to raw difference if peak formula ill-defined
            y[m] = amp * (np.exp(-ts / td) - np.exp(-ts / tr))
        else:
            t_opt = tr * td / (td - tr) * np.log(td / tr)
            norm = np.exp(-t_opt / td) - np.exp(-t_opt / tr)
            rise = np.exp(-ts / tr)
            decay = np.exp(-ts / td)
            y[m] = amp * (decay - rise) / (norm if abs(norm) > 1e-10 else 1.0)
    return y


def model_cooperative_binding(t, amp, tau_rise, tau_decay, n_coop, t_onset):
    """Cooperative binding: Hill-like rise times exponential decay.

    Parameters in seconds (taus) and dimensionless n_coop; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tr = max(tau_rise, 1e-6)
        td = max(tau_decay, 1e-6)
        n = max(n_coop, 0.5)
        x = ts / tr
        rise = (x ** n) / (1.0 + x ** n)
        decay = np.exp(-ts / td)
        y[m] = amp * rise * decay
    return y


def model_two_step_binding(t, amp, tau_bind, tau_conform, tau_dissoc, t_onset):
    """Two-step binding model for iGluSnFR: binding → conformational change → dissociation.
    
    This model captures the biophysical mechanism where conformational change is rate-limiting.
    Uses difference of exponentials to represent sequential binding and conformational change,
    followed by overall dissociation kinetics.
    
    Parameters in seconds; t in milliseconds.
    More accurate than cooperative model for iGluSnFR biophysics.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tb = max(tau_bind, 1e-6)      
        tc = max(tau_conform, 1e-6)   
        td = max(tau_dissoc, 1e-6)    
        
        if tc <= tb:
            tc = tb * 2.0
            
        # Correct sequential kinetics formula
        rise = (np.exp(-ts / tb) - np.exp(-ts / tc)) / (tc - tb)
        
        # Optional: normalize so peak amplitude = 1 before applying amp
        peak_rise_t = tb * tc / (tc - tb) * np.log(tc / tb)
        if peak_rise_t > 0:
            rise_max = (np.exp(-peak_rise_t / tb) - np.exp(-peak_rise_t / tc)) / (tc - tb)
            rise = rise / rise_max
        
        dissoc = np.exp(-ts / td)
        y[m] = amp * rise * dissoc
    return y


def model_single_exp_constrained(t, amp, tau_decay, t_onset):
    """Single exponential decay: instantaneous rise followed by exponential decay.
    
    Biological context: Models processes with instantaneous neurotransmitter release
    and simple first-order clearance kinetics. Applicable to:
    - Fast calcium transients in small compartments
    - Situations where rise time << decay time (e.g., flash photolysis)
    - Simple clearance-dominated kinetics
    
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        y[m] = amp * np.exp(-ts / max(tau_decay, 1e-6))
    return y


def model_alpha_constrained(t, amp, tau, t_onset):
    """Alpha function: (t/τ) * exp(-t/τ). Classic model for synaptic currents.
    
    Biological context: Originally developed to model miniature synaptic currents.
    Represents processes where:
    - Rise and decay are governed by the same time constant
    - Natural for single-pool vesicle fusion kinetics
    - Used in computational neuroscience for simplified synaptic modeling
    - Good for EPSC/IPSC waveforms at room temperature
    
    Parameters in seconds; t in milliseconds.
    Normalized so peak amplitude equals 'amp'.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tau_s = max(tau, 1e-6)
        e_inv = 1.0 / np.e
        y[m] = amp * (ts / tau_s) * np.exp(-ts / tau_s) / e_inv
    return y


def model_gamma_constrained(t, amp, n, tau, t_onset):
    """Gamma function: (t/τ)^n * exp(-t/τ). Models multi-step processes.
    
    Biological context: Represents cascaded processes with multiple rate-limiting steps:
    - Vesicle priming through multiple states before fusion
    - Multi-step enzymatic cascades (e.g., calcium-calmodulin-kinase activation)
    - Sequential binding events in receptor activation
    - Calcium release through multiple coupled stores
    - Higher 'n' values create more delayed, sharper peaks
    
    Parameters in seconds (tau) and dimensionless (n); t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tau_s = max(tau, 1e-6)
        n_safe = max(n, 0.1)
        x = ts / tau_s
        try:
            norm = (n_safe / np.e) ** n_safe
            y[m] = amp * (x ** n_safe) * np.exp(-x) / norm
        except Exception:
            y[m] = amp * (x ** n_safe) * np.exp(-x)
    return y


def model_bilinear_constrained(t, amp, t_rise, t_decay, t_onset):
    """Bilinear rise + exponential decay: linear rise to peak, then exponential decay.
    
    Biological context: Models processes with rate-limited buildup:
    - Sustained neurotransmitter release (e.g., during depolarization)
    - Gradual calcium accumulation in large compartments
    - Slow sensor saturation followed by clearance
    - Useful when rise phase is approximately linear rather than exponential
    
    t_rise and t_decay in milliseconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    rise_mask = (t >= t_onset) & (t <= t_onset + t_rise)
    if np.any(rise_mask):
        t_rel = t[rise_mask] - t_onset
        y[rise_mask] = amp * (t_rel / max(t_rise, 0.1))
    decay_mask = t > (t_onset + t_rise)
    if np.any(decay_mask):
        ts = (t[decay_mask] - t_onset - t_rise) / 1000.0  # Convert to seconds
        tau_s = max(t_decay / 1000.0, 1e-6)  # t_decay is in ms, convert to seconds
        y[decay_mask] = amp * np.exp(-ts / tau_s)
    return y


def model_binding_kinetics(t, amp, kon, koff, tau_clear, t_onset):
    """Binding kinetics with clearance: models receptor binding and dissociation.
    
    Biological context: Explicit modeling of neurotransmitter-receptor interactions:
    - kon: association rate constant (neurotransmitter binding to receptor)
    - koff: dissociation rate constant (neurotransmitter unbinding)
    - tau_clear: additional clearance time constant (uptake, diffusion, metabolism)
    - Useful for iGluSnFR, calcium indicators binding to Ca2+
    - Can model competitive binding scenarios
    
    kon, koff in s⁻¹; tau_clear in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        kon_s = max(kon, 1.0)
        koff_s = max(koff, 1.0)
        binding = 1 - np.exp(-ts * kon_s)
        decay = np.exp(-ts * (koff_s + 1.0 / max(tau_clear, 1e-6)))
        y[m] = amp * binding * decay
    return y


def model_two_component_shared_rise(t, amp_fast, tau_rise, tau_fast, amp_slow, tau_slow, t_onset):
    """Two-component decay with shared rise time: models heterogeneous populations.
    
    Biological context: Represents mixed populations with different kinetics:
    - Fast and slow calcium binding sites with same loading kinetics
    - Mixed receptor populations (e.g., synaptic vs extrasynaptic receptors)
    - Vesicle populations with different release probabilities
    - Buffered vs unbuffered calcium dynamics
    - Different clearance pathways (fast reuptake vs slow diffusion)
    
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        rise = 1 - np.exp(-ts / max(tau_rise, 1e-6))
        fast = amp_fast * np.exp(-ts / max(tau_fast, 1e-6))
        slow = amp_slow * np.exp(-ts / max(tau_slow, 1e-6))
        y[m] = rise * (fast + slow)
    return y


def model_desensitization(t, amp, tau_rise, tau_decay, tau_recovery, desens_factor, t_onset):
    """Desensitization model: activation with progressive reduction due to inactivation.
    
    Biological context: Models receptor or channel desensitization:
    - AMPA/NMDA receptor desensitization during prolonged glutamate exposure
    - Voltage-gated channel inactivation
    - Calcium sensor desensitization (e.g., calmodulin saturation)
    - Progressive reduction in response during sustained stimulation
    - desens_factor: fraction of response lost to desensitization (0-1)
    
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        rise = 1 - np.exp(-ts / max(tau_rise, 1e-6))
        decay = np.exp(-ts / max(tau_decay, 1e-6))
        rec = 1 - desens_factor * (1 - np.exp(-ts / max(tau_recovery, 1e-6)))
        y[m] = amp * rise * decay * rec
    return y


def model_cooperative_plus_linear(t, amp_coop, tau_rise_coop, tau_decay_coop, n_coop, amp_linear, tau_decay_linear, t_onset):
    """Cooperative + linear components: combines Hill-like and simple exponential kinetics.
    
    Biological context: Models mixed binding mechanisms:
    - Cooperative component: high-affinity sites with positive cooperativity
    - Linear component: low-affinity, non-cooperative binding sites  
    - Useful for calcium indicators with multiple binding modes
    - Can represent synaptic + extrasynaptic receptor populations
    - Models situations with both specific and non-specific binding
    
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tr = max(tau_rise_coop, 1e-6)
        n = max(n_coop, 0.5)
        norm_t = ts / tr
        rise_coop = (norm_t ** n) / (1 + norm_t ** n)
        decay_coop = np.exp(-ts / max(tau_decay_coop, 1e-6))
        coop = amp_coop * rise_coop * decay_coop
        rise_linear = 1 - np.exp(-ts / tr)
        decay_linear = np.exp(-ts / max(tau_decay_linear, 1e-6))
        linear = amp_linear * rise_linear * decay_linear
        y[m] = coop + linear
    return y


def model_diffusion_clearance(t, amp, tau_diff, tau_clear1, tau_clear2, frac_clear1, t_onset):
    """Diffusion-limited rise with bi-exponential clearance.
    
    Biological context: Models spatially-distributed processes:
    - tau_diff: diffusion time constant for neurotransmitter spread
    - Dual clearance pathways (e.g., reuptake + metabolism)
    - tau_clear1/2: fast reuptake vs slow enzymatic breakdown
    - frac_clear1: fraction going through fast clearance pathway
    - Relevant for glutamate spillover, volume transmission
    - Models situations where diffusion limits signal rise
    
    Parameters in seconds; t in milliseconds.
    Rise follows alpha function (diffusion kinetics).
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tau_d = max(tau_diff, 1e-6)
        rise = (ts / tau_d) * np.exp(-ts / tau_d)
        e_inv = 1.0 / np.e
        rise_norm = rise / e_inv
        clear1 = frac_clear1 * np.exp(-ts / max(tau_clear1, 1e-6))
        clear2 = (1 - frac_clear1) * np.exp(-ts / max(tau_clear2, 1e-6))
        y[m] = amp * rise_norm * (clear1 + clear2)
    return y


def model_double_cooperative(t, amp, tau_rise1, tau_decay1, n1, tau_rise2, tau_decay2, n2, t_onset):
    """Sum of two cooperative binding components with different kinetics.
    
    Biological context: Represents multiple cooperative binding sites:
    - Different calcium sensors with distinct cooperativities
    - Fast vs slow cooperative gating mechanisms  
    - High-affinity (slow) vs low-affinity (fast) cooperative sites
    - Synaptic vs extrasynaptic cooperative receptors
    - Mixed populations of calcium indicators with different n_coop
    
    Equal weighting (50:50) between components.
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tr1 = max(tau_rise1, 1e-6); n1s = max(n1, 0.5)
        tr2 = max(tau_rise2, 1e-6); n2s = max(n2, 0.5)
        x1 = ts / tr1; x2 = ts / tr2
        rise1 = (x1 ** n1s) / (1 + x1 ** n1s)
        rise2 = (x2 ** n2s) / (1 + x2 ** n2s)
        decay1 = np.exp(-ts / max(tau_decay1, 1e-6))
        decay2 = np.exp(-ts / max(tau_decay2, 1e-6))
        y[m] = amp * (0.5 * rise1 * decay1 + 0.5 * rise2 * decay2)
    return y


def model_heterogeneous_cooperative(t, amp, tau_rise1, tau_decay1, n1, frac1, tau_rise2, tau_decay2, n2, t_onset):
    """Weighted sum of two cooperative components with adjustable fractions.
    
    Biological context: Heterogeneous receptor/sensor populations:
    - frac1: fraction of sites with kinetics 1 vs kinetics 2
    - Models developmental changes in receptor composition
    - Activity-dependent shifts between receptor populations
    - Different splice variants with distinct cooperativities
    - Cell-type specific receptor expression patterns
    
    More flexible than double_cooperative with adjustable weighting.
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        tr1 = max(tau_rise1, 1e-6); tr2 = max(tau_rise2, 1e-6)
        n1s = max(n1, 0.5); n2s = max(n2, 0.5)
        x1 = ts / tr1; x2 = ts / tr2
        rise1 = (x1 ** n1s) / (1 + x1 ** n1s)
        rise2 = (x2 ** n2s) / (1 + x2 ** n2s)
        decay1 = np.exp(-ts / max(tau_decay1, 1e-6))
        decay2 = np.exp(-ts / max(tau_decay2, 1e-6))
        comp1 = frac1 * rise1 * decay1
        comp2 = (1 - frac1) * rise2 * decay2
        y[m] = amp * (comp1 + comp2)
    return y


def model_two_component_cooperative(t, amp_fast, tau_rise_fast, tau_decay_fast, n_fast,
                                    amp_slow, tau_rise_slow, tau_decay_slow, n_slow, t_onset):
    """Two cooperative components with independent amplitudes and kinetics.
    
    Biological context: Most flexible model for heterogeneous cooperative systems:
    - Independent amplitude scaling for each component
    - Different cooperativities for fast vs slow processes
    - Models distinct functional pools (e.g., RRP vs reserve pool vesicles)
    - Synaptic vs extrasynaptic receptors with different cooperativities
    - Useful for calcium indicators in different cellular compartments
    
    Most parameters (9) - use only when simpler models inadequate.
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        # fast component
        trf = max(tau_rise_fast, 1e-6); nf = max(n_fast, 0.5)
        xf = ts / trf
        rise_f = (xf ** nf) / (1 + xf ** nf)
        decay_f = np.exp(-ts / max(tau_decay_fast, 1e-6))
        comp_f = amp_fast * rise_f * decay_f
        # slow component
        trs = max(tau_rise_slow, 1e-6); ns = max(n_slow, 0.5)
        xs = ts / trs
        rise_s = (xs ** ns) / (1 + xs ** ns)
        decay_s = np.exp(-ts / max(tau_decay_slow, 1e-6))
        comp_s = amp_slow * rise_s * decay_s
        y[m] = comp_f + comp_s
    return y

def model_iglusnfr(t, amp, tau_rise, tau_decay_fast, tau_decay_slow, frac_fast, t_onset):
    """Optimized iGluSnFR S72A model with single rise and bi-exponential decay.
    
    Biophysical basis:
    - Single effective rise time constant (combines binding + conformational change)
    - Bi-exponential decay captures heterogeneous unbinding kinetics:
        * Fast component: sensors in partially bound/accessible states
        * Slow component: sensors in fully closed/stable fluorescent states
    
    This formulation is more flexible for fitting both very fast (τ_rise < 2ms) 
    and slower (τ_rise ~ 5-10ms) transients while maintaining interpretability.
    
    Parameters in seconds; t in milliseconds.
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        
        # Ensure valid time constants
        tr = max(tau_rise, 1e-6)
        tdf = max(tau_decay_fast, 1e-6)
        tds = max(tau_decay_slow, 1e-6)
        
        # Ensure decay ordering
        if tdf >= tds:
            tdf = tds * 0.5
        
        # Clamp fraction
        f_fast = np.clip(frac_fast, 0.0, 1.0)
        
        # Single exponential rise (fast rise for S72A)
        rise = 1.0 - np.exp(-ts / tr)
        
        # Bi-exponential decay
        decay = f_fast * np.exp(-ts / tdf) + (1 - f_fast) * np.exp(-ts / tds)
        
        y[m] = amp * rise * decay
    
    return y


def model_iglusnfr_tri(t, amp, tau_rise, tau_decay_fast, tau_decay_slow, tau_decay_superslow,
                       frac_fast, frac_slow, t_onset):
    """Tri-exponential iGluSnFR model with fast, slow, and super-slow decay components.
    
    Biophysical basis:
    - Single effective rise time constant (binding + initial conformational change)
    - Tri-exponential decay captures full heterogeneous kinetics:
        * Fast (1-10ms): intrinsic sensor unbinding from high-affinity state
        * Slow (10-25ms): conformational relaxation / intermediate states
        * Super-slow (25-150ms): glutamate accumulation / spillover effects
          (typically fixed from post-train decay fitting)
    
    The super-slow component builds up during train stimulation and dominates
    the post-train decay, while fast/slow components dominate early events.
    
    Parameters in seconds; t in milliseconds.
    frac_fast + frac_slow <= 1.0; frac_superslow = 1 - frac_fast - frac_slow
    """
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    m = t >= t_onset
    
    if np.any(m):
        ts = (t[m] - t_onset) / 1000.0
        
        # Ensure valid time constants
        tr = max(tau_rise, 1e-6)
        tdf = max(tau_decay_fast, 1e-6)
        tds = max(tau_decay_slow, 1e-6)
        tdss = max(tau_decay_superslow, 1e-6)
        
        # Ensure decay ordering: fast < slow < superslow
        if tdf >= tds:
            tdf = tds * 0.5
        if tds >= tdss:
            tds = tdss * 0.5
        
        # Clamp fractions to valid range
        f_fast = np.clip(frac_fast, 0.0, 1.0)
        f_slow = np.clip(frac_slow, 0.0, 1.0 - f_fast)
        f_superslow = 1.0 - f_fast - f_slow
        
        # Single exponential rise
        rise = 1.0 - np.exp(-ts / tr)
        
        # Tri-exponential decay
        decay = (f_fast * np.exp(-ts / tdf) + 
                 f_slow * np.exp(-ts / tds) + 
                 f_superslow * np.exp(-ts / tdss))
        
        y[m] = amp * rise * decay
    
    return y


def get_event_model(name: str) -> Dict:
    """Return a model spec dict for the given name.

    Dict keys:
    - 'func': callable(t_ms, *params)
    - 'params': list of parameter names
    - 'bounds': (lb, ub) passed to curve_fit
    - 'p0_func': callable(y_fit, t_fit) -> initial params
    - 'name': canonical name
    """
    nm = (name or '').strip().lower()
    if nm in ('double', 'double_exp', 'double-exponential', 'biexp'):
        return _apply_global_bounds({
            'name': 'double_exp',
            'func': model_double_exp_constrained,
            'params': ['amp', 'tau_rise', 'tau_decay', 't_onset'],
            'bounds': ([0, 0.0005, 0.001, 0], [np.inf, 0.010, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.002, 0.020, float(t[np.nanargmax(y)])],
            'complexity': 4,
            'progression_rules': {
                'tau_rise': 'monotonic_increasing',
                'tau_decay': 'monotonic_increasing',
                'amp': 'free',
                't_onset': 'free',
            },
        })
    if nm in ('coop', 'cooperative', 'cooperative_binding'):
        return _apply_global_bounds({
            'name': 'cooperative',
            'func': model_cooperative_binding,
            'params': ['amp', 'tau_rise', 'tau_decay', 'n_coop', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0], [np.inf, 0.020, 0.200, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.005, 0.030, 2.0, float(t[np.nanargmax(y)])],
            'complexity': 5,
        })
    if nm in ('two_step', 'two_step_binding', 'two-step', 'iglusnfr_biophysical'):
            return _apply_global_bounds({
                'name': 'two_step_binding',
                'func': model_two_step_binding,
                'params': ['amp', 'tau_bind', 'tau_conform', 'tau_dissoc', 't_onset'],
                'bounds': ([0, 0.0001, 0.0008, 0.005, 0], [np.inf, 0.002, 0.015, 0.300, 10]),
                'p0_func': lambda y, t: [float(np.nanmax(y)), 0.0005, 0.003, 0.040, float(t[np.nanargmax(y)])],
                'complexity': 5,
            })
    if nm in ('iglusnfr', 'iglusnfr_s72a', 'markov_4state'):
        return _apply_global_bounds({
            'name': 'iglusnfr',
            'func': model_iglusnfr,
            'params': ['amp', 'tau_rise', 'tau_decay_fast', 'tau_decay_slow', 'frac_fast', 't_onset'],
            # Widened bounds to capture full range of S72A kinetics
            # tau_rise: 0.1-15 ms (very fast to moderate)
            # tau_decay_fast: 2-30 ms (fast unbinding)
            # tau_decay_slow: 8-150 ms (slow dissociation)
            # frac_fast: 0.1-0.9 (flexible weighting)
            'bounds': (
                [0,      0.0001, 0.002,  0.008,  0.1,   0],      # lower bounds
                [np.inf, 0.015,  0.030,  0.150,  0.9,   10]      # upper bounds
            ),
            'p0_func': lambda y, t: [
                float(np.nanmax(y)),                    # amp
                0.003,                                   # tau_rise: 3 ms
                0.008,                                   # tau_decay_fast: 8 ms
                0.035,                                   # tau_decay_slow: 35 ms
                0.6,                                     # frac_fast: 60%
                float(np.clip(t[np.nanargmax(y)], 0, 10))  # t_onset (clipped to bounds for robustness)
            ],
            'complexity': 6,
            # Per-parameter progression rules for train dynamics
            # 'monotonic_increasing': parameter can only increase (e.g., tau gets slower)
            # 'monotonic_decreasing': parameter can only decrease
            # 'free': no monotonic constraint (default if not specified)
            'progression_rules': {
                'tau_rise': 'monotonic_increasing',         # glutamate accumulation → slower rise
                'tau_decay_fast': 'monotonic_increasing',   # glutamate accumulation → slower decay
                'tau_decay_slow': 'monotonic_increasing',   # glutamate accumulation → slower decay
                'frac_fast': 'free',                        # can vary either direction
                'slow_fraction': 'monotonic_increasing',    # slow component fraction increases across train
                'amp': 'free',                              # can increase or decrease
                't_onset': 'free',                           # timing parameter
            },
        })
    if nm in ('iglusnfr_tri', 'iglusnfr_triexp', 'triexp'):
        return _apply_global_bounds({
            'name': 'iglusnfr_tri',
            'func': model_iglusnfr_tri,
            'params': ['amp', 'tau_rise', 'tau_decay_fast', 'tau_decay_slow', 'tau_decay_superslow',
                       'frac_fast', 'frac_slow', 't_onset'],
            # Tri-exponential with NON-OVERLAPPING ranges:
            # tau_rise: 0.5-5 ms
            # tau_decay_fast: 1-10 ms (intrinsic unbinding)
            # tau_decay_slow: 10-25 ms (conformational/intermediate)
            # tau_decay_superslow: 25-150 ms (accumulation, typically fixed from post-train)
            'bounds': (
                [0,      0.0005, 0.001,  0.010,  0.025,  0.1,   0.1,   0],      # lower bounds
                [np.inf, 0.005,  0.010,  0.025,  0.150,  0.7,   0.6,   10]      # upper bounds
            ),
            'p0_func': lambda y, t: [
                float(np.nanmax(y)),                    # amp
                0.002,                                   # tau_rise: 2 ms
                0.005,                                   # tau_decay_fast: 5 ms
                0.015,                                   # tau_decay_slow: 15 ms
                0.040,                                   # tau_decay_superslow: 40 ms
                0.5,                                     # frac_fast: 50%
                0.3,                                     # frac_slow: 30% (superslow = 20%)
                float(np.clip(t[np.nanargmax(y)], 0, 10))
            ],
            'complexity': 8,
            'progression_rules': {
                'tau_rise': 'monotonic_increasing',
                'tau_decay_fast': 'monotonic_increasing',
                'tau_decay_slow': 'monotonic_increasing',
                'tau_decay_superslow': 'free',              # Fixed from post-train decay
                'frac_fast': 'monotonic_decreasing',        # Less fast component as train progresses
                'frac_slow': 'free',
                'amp': 'free',
                't_onset': 'free',
            },
        })
    if nm in ('single', 'single_exp', 'single-exponential'):
        return _apply_global_bounds({
            'name': 'single_exp',
            'func': model_single_exp_constrained,
            'params': ['amp', 'tau_decay', 't_onset'],
            'bounds': ([0, 0.001, 0], [np.inf, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.020, float(t[np.nanargmax(y)])],
            'complexity': 3,
        })
    if nm in ('alpha',):
        return _apply_global_bounds({
            'name': 'alpha',
            'func': model_alpha_constrained,
            'params': ['amp', 'tau', 't_onset'],
            'bounds': ([0, 0.001, 0], [np.inf, 0.100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)) * np.e, 0.010, float(t[np.nanargmax(y)])],
            'complexity': 3,
        })
    if nm in ('gamma',):
        return _apply_global_bounds({
            'name': 'gamma',
            'func': model_gamma_constrained,
            'params': ['amp', 'n', 'tau', 't_onset'],
            'bounds': ([0, 0.5, 0.001, 0], [np.inf, 8.0, 0.100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)) * 3.0, 2.0, 0.010, float(t[np.nanargmax(y)])],
            'complexity': 4,
        })
    if nm in ('bilinear',):
        return _apply_global_bounds({
            'name': 'bilinear',
            'func': model_bilinear_constrained,
            'params': ['amp', 't_rise', 't_decay', 't_onset'],
            'bounds': ([0, 0.1, 1, 0], [np.inf, 10, 100, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 2.0, 20.0, float(t[np.nanargmax(y)])],
            'complexity': 4,
        })
    if nm in ('binding_kinetics', 'binding'):
        return _apply_global_bounds({
            'name': 'binding_kinetics',
            'func': model_binding_kinetics,
            'params': ['amp', 'kon', 'koff', 'tau_clear', 't_onset'],
            'bounds': ([0, 10, 1, 0.001, 0], [np.inf, 1000, 200, 0.200, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 200, 50, 0.030, float(t[np.nanargmax(y)])],
            'complexity': 5,
        })
    if nm in ('two_component', 'two-component', 'two_component_shared_rise'):
        return _apply_global_bounds({
            'name': 'two_component',
            'func': model_two_component_shared_rise,
            'params': ['amp_fast', 'tau_rise', 'tau_fast', 'amp_slow', 'tau_slow', 't_onset'],
            'bounds': ([0, 0.0005, 0.001, 0, 0.010, 0], [np.inf, 0.010, 0.100, np.inf, 1.000, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.6, 0.002, 0.015, float(np.nanmax(y))*0.4, 0.080, float(t[np.nanargmax(y)])],
            'complexity': 6,
        })
    if nm in ('desens', 'desensitization'):
        return _apply_global_bounds({
            'name': 'desensitization',
            'func': model_desensitization,
            'params': ['amp', 'tau_rise', 'tau_decay', 'tau_recovery', 'desens_factor', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.020, 0, 0], [np.inf, 0.010, 0.100, 1.000, 0.8, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.020, 0.100, 0.3, float(t[np.nanargmax(y)])],
            'complexity': 6,
        })
    if nm in ('coop_plus_linear', 'cooperative_plus_linear'):
        return _apply_global_bounds({
            'name': 'coop_plus_linear',
            'func': model_cooperative_plus_linear,
            'params': ['amp_coop', 'tau_rise_coop', 'tau_decay_coop', 'n_coop', 'amp_linear', 'tau_decay_linear', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0, 0.010, 0], [np.inf, 0.020, 0.200, 5.0, np.inf, 0.500, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.8, 0.005, 0.030, 2.0, float(np.nanmax(y))*0.2, 0.100, float(t[np.nanargmax(y)])],
            'complexity': 7,
        })
    if nm in ('diffusion_clearance', 'diffusion'):
        return _apply_global_bounds({
            'name': 'diffusion_clearance',
            'func': model_diffusion_clearance,
            'params': ['amp', 'tau_diff', 'tau_clear1', 'tau_clear2', 'frac_clear1', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.020, 0.1, 0], [np.inf, 0.020, 0.100, 0.500, 0.9, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*np.e, 0.003, 0.015, 0.080, 0.6, float(t[np.nanargmax(y)])],
            'complexity': 7,
        })
    if nm in ('double_cooperative', 'double_coop'):
        return _apply_global_bounds({
            'name': 'double_cooperative',
            'func': model_double_cooperative,
            'params': ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'tau_rise2', 'tau_decay2', 'n2', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.100, 5.0, 0.100, 0.500, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.015, 2.0, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 8,
        })
    if nm in ('hetero_coop', 'heterogeneous_cooperative'):
        return _apply_global_bounds({
            'name': 'hetero_coop',
            'func': model_heterogeneous_cooperative,
            'params': ['amp', 'tau_rise1', 'tau_decay1', 'n1', 'frac1', 'tau_rise2', 'tau_decay2', 'n2', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0.1, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.200, 5.0, 0.9, 0.100, 1.000, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y)), 0.003, 0.020, 2.0, 0.6, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 9,
        })
    if nm in ('two_comp_coop', 'two_component_cooperative'):
        return _apply_global_bounds({
            'name': 'two_comp_coop',
            'func': model_two_component_cooperative,
            'params': ['amp_fast', 'tau_rise_fast', 'tau_decay_fast', 'n_fast', 'amp_slow', 'tau_rise_slow', 'tau_decay_slow', 'n_slow', 't_onset'],
            'bounds': ([0, 0.001, 0.005, 0.5, 0, 0.005, 0.020, 0.5, 0], [np.inf, 0.020, 0.100, 5.0, np.inf, 0.100, 1.000, 5.0, 10]),
            'p0_func': lambda y, t: [float(np.nanmax(y))*0.6, 0.003, 0.015, 2.0, float(np.nanmax(y))*0.4, 0.010, 0.080, 1.5, float(t[np.nanargmax(y)])],
            'complexity': 9,
        })
    raise ValueError(f"Unknown model '{name}'")


def get_models(names: List[str]) -> Dict[str, Dict]:
    out = {}
    for n in names:
        spec = get_event_model(n)
        out[spec['name']] = spec
    return out
