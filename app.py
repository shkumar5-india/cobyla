import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock

# --- UI CONFIG ---
st.set_page_config(page_title="QUANTUM VQE LAB | COBYLA", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e293b; text-align: center; margin-bottom: 30px; }
    .metric-card {
        background: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center;
    }
    .spec-label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; font-weight: 600; }
    .spec-value { color: #0f172a; font-size: 1.5rem; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🧪 COBYLA QUANTUM MOLECULAR SOLVER</div>', unsafe_allow_html=True)

def execute_vqe_engine(mol_type):
    # Setup parameters based on molecule selection
    if mol_type == "H2":
        d_range = np.arange(0.5, 2.1, 0.4)
        mapper = ParityMapper(num_particles=(1, 1))
        atom_symbol = "H"
        use_active_space = False
    else: # LiH
        d_range = [1.2, 1.5, 1.8, 2.1]
        mapper = JordanWignerMapper()
        atom_symbol = "Li"
        use_active_space = True

    # Initialize COBYLA Optimizer
    # rhobeg: initial step size, maxiter: budget for evaluations
    optimizer = COBYLA(maxiter=200, rhobeg=0.1)

    all_dist, all_energ = [], []
    best_overall_energy = float("inf")
    best_conv_history = []
    best_ansatz = None
    
    status = st.status(f"Solving {mol_type} with COBYLA...", expanded=True)

    for d in d_range:
        status.update(label=f"Optimizing at Distance: {d:.2f} Å", state="running")
        
        # 1. Chemical Problem Setup
        driver = PySCFDriver(atom=f"{atom_symbol} 0 0 0; H 0 0 {d}", basis="sto3g")
        problem = driver.run()
        
        # Capture constant energy shifts (Nuclear + Core)
        energy_shift = sum(problem.hamiltonian.constants.values())
        ham_op = problem.hamiltonian.second_q_op()
        
        # 2. Complexity Reduction (for LiH)
        if use_active_space:
            transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
            problem = transformer.transform(problem)
            ham_op = problem.hamiltonian.second_q_op()
            energy_shift = sum(problem.hamiltonian.constants.values())

        # 3. VQE Components
        ansatz = UCCSD(
            problem.num_spatial_orbitals, problem.num_particles, mapper,
            initial_state=HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
        )

        temp_history = []
        def callback(count, params, mean, std=None): 
            temp_history.append(mean + energy_shift)

        # 4. Run VQE
        vqe = VQE(Estimator(), ansatz, optimizer, callback=callback)
        result = vqe.compute_minimum_eigenvalue(mapper.map(ham_op))

        total_energy = float(result.eigenvalue.real) + energy_shift
        all_dist.append(d)
        all_energ.append(total_energy)

        # Record data for the equilibrium point
        if total_energy < best_overall_energy:
            best_overall_energy = total_energy
            best_conv_history = temp_history
            best_ansatz = ansatz

    status.update(label="VQE Complete", state="complete", expanded=False)
    hw_circ = transpile(best_ansatz, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=1)
    
    return all_dist, all_energ, best_conv_history, all_dist[np.argmin(all_energ)], best_overall_energy, hw_circ

# --- UI CONTROL FLOW ---
with st.sidebar:
    st.header("VQE Configuration")
    mol_choice = st.selectbox("Select Target", ["H2", "LiH"])
    st.info("Using COBYLA Optimizer: A gradient-free algorithm often used in noisy quantum simulations.")

run_btn = st.button("RUN COBYLA ANALYSIS", use_container_width=True, type="primary")

if run_btn:
    d, e, conv, b_dist, m_e, circ = execute_vqe_engine(mol_choice)
    
    # KPI METRICS
    st.markdown("### 📊 QUANTUM PERFORMANCE METRICS")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><p class="spec-label">Eq. Distance</p><p class="spec-value">{b_dist:.3f} Å</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><p class="spec-label">Ground State</p><p class="spec-value">{m_e:.6f} Ha</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><p class="spec-label">COBYLA Evals</p><p class="spec-value">{len(conv)}</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><p class="spec-label">Logical Qubits</p><p class="spec-value">{circ.num_qubits}</p></div>', unsafe_allow_html=True)

    st.divider()
    
    # VISUALIZATION
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(d, e, 's--', color='#10b981', label="COBYLA PES")
        ax1.set_title("Potential Energy Surface (PES)", fontweight='bold')
        ax1.set_xlabel("Interatomic Distance (Å)"); ax1.set_ylabel("Total Energy (Ha)")
        ax1.grid(alpha=0.2); ax1.legend()
        st.pyplot(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(conv, color='#f59e0b', lw=2)
        ax2.set_title("Optimizer Convergence Path", fontweight='bold')
        ax2.set_xlabel("Function Evaluations"); ax2.set_ylabel("Energy (Ha)")
        ax2.grid(alpha=0.2)
        st.pyplot(fig2)

    with st.expander("🛠 HARDWARE & CIRCUIT DIAGNOSTICS"):
        st.markdown(f"**Optimizer:** COBYLA (maxiter=200)")
        st.markdown(f"**Circuit Depth:** {circ.depth()}")
        st.table({"Gate Type": list(circ.count_ops().keys()), "Count": list(circ.count_ops().values())})
        st.pyplot(circ.draw('mpl', scale=0.8))

else:
    st.write("Click the button above to start the simulation.")
