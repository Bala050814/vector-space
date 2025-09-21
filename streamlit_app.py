import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
from fractions import Fraction
import sympy as sp
from sympy import symbols, Matrix, simplify, latex

def vector_addition(v1, v2):
    """Add two vectors"""
    return np.array(v1) + np.array(v2)

def scalar_multiplication(scalar, vector):
    """Multiply vector by scalar"""
    return scalar * np.array(vector)

def dot_product(v1, v2):
    """Calculate dot product of two vectors"""
    return np.dot(v1, v2)

def cross_product(v1, v2):
    """Calculate cross product of two 3D vectors"""
    if len(v1) != 3 or len(v2) != 3:
        return None
    return np.cross(v1, v2)

def vector_magnitude(vector):
    """Calculate magnitude (norm) of a vector"""
    return np.linalg.norm(vector)

def vector_normalize(vector):
    """Normalize a vector"""
    magnitude = vector_magnitude(vector)
    if magnitude == 0:
        return vector
    return np.array(vector) / magnitude

def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    dot_prod = dot_product(v1, v2)
    mag_v1 = vector_magnitude(v1)
    mag_v2 = vector_magnitude(v2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    
    cos_angle = dot_prod / (mag_v1 * mag_v2)
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def project_vector(v1, v2):
    """Project vector v1 onto vector v2"""
    dot_prod = dot_product(v1, v2)
    mag_v2_sq = dot_product(v2, v2)
    
    if mag_v2_sq == 0:
        return np.zeros_like(v1)
    
    return (dot_prod / mag_v2_sq) * np.array(v2)

def are_linearly_independent(vectors):
    """Check if vectors are linearly independent"""
    if not vectors:
        return False
    
    # Create matrix with vectors as columns
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    
    return rank == len(vectors)

def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization process"""
    if not vectors:
        return []
    
    orthogonal_vectors = []
    
    for i, v in enumerate(vectors):
        v = np.array(v, dtype=float)
        
        # Subtract projections onto previous orthogonal vectors
        for orth_v in orthogonal_vectors:
            projection = project_vector(v, orth_v)
            v = v - projection
        
        # Check if vector is non-zero (linearly independent)
        if vector_magnitude(v) > 1e-10:
            orthogonal_vectors.append(v)
    
    return orthogonal_vectors

def gram_schmidt_orthonormal(vectors):
    """Gram-Schmidt process with normalization"""
    orthogonal = gram_schmidt(vectors)
    return [vector_normalize(v) for v in orthogonal]

def vector_span_basis(vectors):
    """Find basis for the span of given vectors"""
    if not vectors:
        return []
    
    # Create matrix and find RREF
    matrix = np.array(vectors).T
    rank = np.linalg.matrix_rank(matrix)
    
    # Use QR decomposition to find basis
    Q, R = np.linalg.qr(matrix)
    
    # Extract linearly independent columns
    basis_vectors = []
    for i in range(min(rank, matrix.shape[1])):
        if i < matrix.shape[1]:
            basis_vectors.append(matrix[:, i])
    
    return basis_vectors

def plot_2d_vectors(vectors, labels=None, colors=None):
    """Plot 2D vectors"""
    fig = go.Figure()
    
    if colors is None:
        colors = px.colors.qualitative.Set1
    
    for i, vector in enumerate(vectors):
        if len(vector) >= 2:
            label = labels[i] if labels and i < len(labels) else f"Vector {i+1}"
            color = colors[i % len(colors)]
            
            # Add vector arrow
            fig.add_trace(go.Scatter(
                x=[0, vector[0]], 
                y=[0, vector[1]],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=[0, 10], color=color)
            ))
            
            # Add arrow annotation
            fig.add_annotation(
                x=vector[0], y=vector[1],
                ax=0, ay=0,
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=color,
                showarrow=True
            )
    
    # Set equal aspect ratio
    max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors if len(v) >= 2]) * 1.2
    
    fig.update_layout(
        title="2D Vector Visualization",
        xaxis=dict(range=[-max_val, max_val], title="X"),
        yaxis=dict(range=[-max_val, max_val], title="Y"),
        showlegend=True,
        width=600,
        height=600
    )
    
    return fig

def plot_3d_vectors(vectors, labels=None, colors=None):
    """Plot 3D vectors"""
    fig = go.Figure()
    
    if colors is None:
        colors = px.colors.qualitative.Set1
    
    for i, vector in enumerate(vectors):
        if len(vector) >= 3:
            label = labels[i] if labels and i < len(labels) else f"Vector {i+1}"
            color = colors[i % len(colors)]
            
            # Add vector line
            fig.add_trace(go.Scatter3d(
                x=[0, vector[0]], 
                y=[0, vector[1]], 
                z=[0, vector[2]],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=6),
                marker=dict(size=[0, 8], color=color)
            ))
    
    fig.update_layout(
        title="3D Vector Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        showlegend=True,
        width=700,
        height=600
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Vector Spaces Calculator",
        page_icon="🔢",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .operation-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
    .result-box {
        background-color: #e7f3ff;
        border: 2px solid #007bff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .formula-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔢 Vector Spaces Calculator</h1>
        <p>Comprehensive tool for vector operations and linear algebra</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "➕ Basic Operations", 
        "📐 Geometric Properties", 
        "🔗 Linear Combinations", 
        "⊥ Orthogonalization", 
        "📊 Visualization", 
        "📚 Theory"
    ])
    
    with tab1:
        st.header("➕ Basic Vector Operations")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Input Vectors")
            
            # Dimension selection
            dim = st.selectbox("Vector Dimension:", [2, 3, 4, 5], index=1)
            
            # Vector A input
            st.write("**Vector A:**")
            vector_a = []
            cols_a = st.columns(dim)
            for i in range(dim):
                with cols_a[i]:
                    val = st.number_input(f"a{i+1}", value=1.0, key=f"a_{i}", format="%.3f")
                    vector_a.append(val)
            
            # Vector B input
            st.write("**Vector B:**")
            vector_b = []
            cols_b = st.columns(dim)
            for i in range(dim):
                with cols_b[i]:
                    val = st.number_input(f"b{i+1}", value=2.0, key=f"b_{i}", format="%.3f")
                    vector_b.append(val)
            
            # Scalar input
            scalar = st.number_input("Scalar (k):", value=2.0, format="%.3f")
        
        with col2:
            st.subheader("📊 Results")
            
            # Vector addition
            result_add = vector_addition(vector_a, vector_b)
            st.markdown("""
            <div class="result-box">
                <h4>Vector Addition: A + B</h4>
                <p><strong>Result:</strong> [{}]</p>
                <p><strong>Formula:</strong> (a₁+b₁, a₂+b₂, ...)</p>
            </div>
            """.format(", ".join([f"{x:.3f}" for x in result_add])), unsafe_allow_html=True)
            
            # Vector subtraction
            result_sub = np.array(vector_a) - np.array(vector_b)
            st.markdown("""
            <div class="result-box">
                <h4>Vector Subtraction: A - B</h4>
                <p><strong>Result:</strong> [{}]</p>
            </div>
            """.format(", ".join([f"{x:.3f}" for x in result_sub])), unsafe_allow_html=True)
            
            # Scalar multiplication
            result_scalar_a = scalar_multiplication(scalar, vector_a)
            st.markdown("""
            <div class="result-box">
                <h4>Scalar Multiplication: k × A</h4>
                <p><strong>Result:</strong> [{}]</p>
                <p><strong>Scalar k:</strong> {}</p>
            </div>
            """.format(", ".join([f"{x:.3f}" for x in result_scalar_a]), scalar), unsafe_allow_html=True)
            
            # Dot product
            dot_prod = dot_product(vector_a, vector_b)
            st.markdown("""
            <div class="result-box">
                <h4>Dot Product: A · B</h4>
                <p><strong>Result:</strong> {:.3f}</p>
                <p><strong>Formula:</strong> Σ(aᵢ × bᵢ)</p>
            </div>
            """.format(dot_prod), unsafe_allow_html=True)
            
            # Cross product (only for 3D)
            if dim == 3:
                cross_prod = cross_product(vector_a, vector_b)
                if cross_prod is not None:
                    st.markdown("""
                    <div class="result-box">
                        <h4>Cross Product: A × B</h4>
                        <p><strong>Result:</strong> [{}]</p>
                        <p><strong>Note:</strong> Perpendicular to both A and B</p>
                    </div>
                    """.format(", ".join([f"{x:.3f}" for x in cross_prod])), unsafe_allow_html=True)
    
    with tab2:
        st.header("📐 Geometric Properties")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Input Vector")
            
            # Use vectors from previous tab or new input
            if 'vector_a' not in locals():
                dim_geom = st.selectbox("Dimension:", [2, 3, 4], index=1, key="geom_dim")
                vector_geom = []
                cols_geom = st.columns(dim_geom)
                for i in range(dim_geom):
                    with cols_geom[i]:
                        val = st.number_input(f"v{i+1}", value=3.0, key=f"geom_{i}", format="%.3f")
                        vector_geom.append(val)
            else:
                vector_geom = vector_a
                dim_geom = len(vector_geom)
            
            # Second vector for angle and projection
            st.write("**Second Vector (for angle/projection):**")
            vector_geom2 = []
            cols_geom2 = st.columns(dim_geom)
            for i in range(dim_geom):
                with cols_geom2[i]:
                    val = st.number_input(f"u{i+1}", value=1.0, key=f"geom2_{i}", format="%.3f")
                    vector_geom2.append(val)
        
        with col2:
            st.subheader("📊 Geometric Analysis")
            
            # Magnitude
            magnitude = vector_magnitude(vector_geom)
            st.markdown("""
            <div class="result-box">
                <h4>Vector Magnitude ||v||</h4>
                <p><strong>Result:</strong> {:.6f}</p>
                <p><strong>Formula:</strong> √(v₁² + v₂² + ... + vₙ²)</p>
            </div>
            """.format(magnitude), unsafe_allow_html=True)
            
            # Unit vector
            if magnitude > 0:
                unit_vector = vector_normalize(vector_geom)
                st.markdown("""
                <div class="result-box">
                    <h4>Unit Vector v̂</h4>
                    <p><strong>Result:</strong> [{}]</p>
                    <p><strong>Magnitude:</strong> {:.6f}</p>
                </div>
                """.format(", ".join([f"{x:.6f}" for x in unit_vector]), 
                          vector_magnitude(unit_vector)), unsafe_allow_html=True)
            
            # Angle between vectors
            angle = angle_between_vectors(vector_geom, vector_geom2)
            st.markdown("""
            <div class="result-box">
                <h4>Angle Between Vectors</h4>
                <p><strong>Degrees:</strong> {:.3f}°</p>
                <p><strong>Radians:</strong> {:.6f}</p>
                <p><strong>Formula:</strong> cos⁻¹((u·v)/(||u||||v||))</p>
            </div>
            """.format(angle, math.radians(angle)), unsafe_allow_html=True)
            
            # Vector projection
            projection = project_vector(vector_geom, vector_geom2)
            proj_magnitude = vector_magnitude(projection)
            st.markdown("""
            <div class="result-box">
                <h4>Projection of v onto u</h4>
                <p><strong>Result:</strong> [{}]</p>
                <p><strong>Magnitude:</strong> {:.6f}</p>
                <p><strong>Formula:</strong> ((v·u)/||u||²) × u</p>
            </div>
            """.format(", ".join([f"{x:.6f}" for x in projection]), proj_magnitude), 
               unsafe_allow_html=True)
            
            # Orthogonal component
            orthogonal = np.array(vector_geom) - projection
            st.markdown("""
            <div class="result-box">
                <h4>Orthogonal Component</h4>
                <p><strong>Result:</strong> [{}]</p>
                <p><strong>Note:</strong> v - proj_u(v)</p>
            </div>
            """.format(", ".join([f"{x:.6f}" for x in orthogonal])), unsafe_allow_html=True)
    
    with tab3:
        st.header("🔗 Linear Combinations & Independence")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Input Vector Set")
            
            # Number of vectors
            num_vectors = st.slider("Number of vectors:", 2, 5, 3)
            dim_linear = st.selectbox("Dimension:", [2, 3, 4], index=1, key="linear_dim")
            
            # Input vectors
            vectors_linear = []
            for i in range(num_vectors):
                st.write(f"**Vector {i+1}:**")
                vector = []
                cols = st.columns(dim_linear)
                for j in range(dim_linear):
                    with cols[j]:
                        val = st.number_input(
                            f"v{i+1}{j+1}", 
                            value=float(i+1 if j == i and i < dim_linear else (1.0 if j == 0 else 0.0)), 
                            key=f"linear_{i}_{j}", 
                            format="%.3f"
                        )
                        vector.append(val)
                vectors_linear.append(vector)
            
            # Coefficients for linear combination
            st.subheader("📊 Linear Combination Coefficients")
            coeffs = []
            coeff_cols = st.columns(num_vectors)
            for i in range(num_vectors):
                with coeff_cols[i]:
                    coeff = st.number_input(f"c{i+1}", value=1.0, key=f"coeff_{i}", format="%.3f")
                    coeffs.append(coeff)
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            # Linear combination
            linear_comb = np.zeros(dim_linear)
            for i, coeff in enumerate(coeffs):
                linear_comb += coeff * np.array(vectors_linear[i])
            
            combination_str = " + ".join([f"{coeffs[i]:.3f}×v{i+1}" for i in range(num_vectors)])
            st.markdown("""
            <div class="result-box">
                <h4>Linear Combination</h4>
                <p><strong>Expression:</strong> {}</p>
                <p><strong>Result:</strong> [{}]</p>
            </div>
            """.format(combination_str, ", ".join([f"{x:.6f}" for x in linear_comb])), 
               unsafe_allow_html=True)
            
            # Linear independence check
            is_independent = are_linearly_independent(vectors_linear)
            independence_color = "success-box" if is_independent else "warning-box"
            independence_text = "✅ Linearly Independent" if is_independent else "❌ Linearly Dependent"
            
            st.markdown(f"""
            <div class="{independence_color}">
                <h4>Linear Independence</h4>
                <p><strong>Status:</strong> {independence_text}</p>
                <p><strong>Explanation:</strong> {'The vectors form a basis for their span.' if is_independent else 'One vector can be written as a combination of others.'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Matrix rank
            if vectors_linear:
                matrix = np.column_stack(vectors_linear)
                rank = np.linalg.matrix_rank(matrix)
                st.markdown("""
                <div class="result-box">
                    <h4>Matrix Properties</h4>
                    <p><strong>Matrix Rank:</strong> {}</p>
                    <p><strong>Number of Vectors:</strong> {}</p>
                    <p><strong>Dimension:</strong> {}</p>
                </div>
                """.format(rank, num_vectors, dim_linear), unsafe_allow_html=True)
            
            # Span basis
            try:
                basis_vectors = gram_schmidt(vectors_linear)
                if basis_vectors:
                    st.markdown("""
                    <div class="result-box">
                        <h4>Orthogonal Basis for Span</h4>
                        <p><strong>Number of basis vectors:</strong> {}</p>
                    </div>
                    """.format(len(basis_vectors)), unsafe_allow_html=True)
                    
                    for i, basis_vec in enumerate(basis_vectors):
                        st.write(f"**Basis Vector {i+1}:** [{', '.join([f'{x:.6f}' for x in basis_vec])}]")
            except:
                st.warning("Could not compute orthogonal basis.")
    
    with tab4:
        st.header("⊥ Gram-Schmidt Orthogonalization")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 Input Vector Set")
            
            num_vectors_gs = st.slider("Number of vectors:", 2, 4, 3, key="gs_num")
            dim_gs = st.selectbox("Dimension:", [2, 3, 4], index=1, key="gs_dim")
            
            vectors_gs = []
            for i in range(num_vectors_gs):
                st.write(f"**Vector {i+1}:**")
                vector = []
                cols = st.columns(dim_gs)
                for j in range(dim_gs):
                    with cols[j]:
                        # Default to somewhat orthogonal vectors
                        default_val = 1.0 if j == i else (0.5 if j < i else 0.0)
                        val = st.number_input(
                            f"u{i+1}{j+1}", 
                            value=default_val,
                            key=f"gs_{i}_{j}", 
                            format="%.3f"
                        )
                        vector.append(val)
                vectors_gs.append(vector)
        
        with col2:
            st.subheader("📊 Orthogonalization Results")
            
            # Gram-Schmidt orthogonalization
            try:
                orthogonal_vectors = gram_schmidt(vectors_gs)
                orthonormal_vectors = gram_schmidt_orthonormal(vectors_gs)
                
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Gram-Schmidt Process Complete</h4>
                    <p><strong>Original vectors:</strong> {}</p>
                    <p><strong>Orthogonal vectors:</strong> {}</p>
                    <p><strong>Orthonormal vectors:</strong> {}</p>
                </div>
                """.format(num_vectors_gs, len(orthogonal_vectors), len(orthonormal_vectors)), 
                   unsafe_allow_html=True)
                
                # Display orthogonal vectors
                st.subheader("🔄 Orthogonal Vectors")
                for i, orth_vec in enumerate(orthogonal_vectors):
                    magnitude = vector_magnitude(orth_vec)
                    st.write(f"**v{i+1}':** [{', '.join([f'{x:.6f}' for x in orth_vec])}] (||v{i+1}'|| = {magnitude:.6f})")
                
                # Display orthonormal vectors
                st.subheader("🎯 Orthonormal Vectors")
                for i, orthonormal_vec in enumerate(orthonormal_vectors):
                    magnitude = vector_magnitude(orthonormal_vec)
                    st.write(f"**e{i+1}:** [{', '.join([f'{x:.6f}' for x in orthonormal_vec])}] (||e{i+1}|| = {magnitude:.6f})")
                
                # Verify orthogonality
                st.subheader("✅ Orthogonality Verification")
                for i in range(len(orthonormal_vectors)):
                    for j in range(i+1, len(orthonormal_vectors)):
                        dot_prod = dot_product(orthonormal_vectors[i], orthonormal_vectors[j])
                        st.write(f"e{i+1} · e{j+1} = {dot_prod:.10f}")
                
            except Exception as e:
                st.error(f"Error in Gram-Schmidt process: {str(e)}")
    
    with tab5:
        st.header("📊 Vector Visualization")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎨 Visualization Settings")
            
            viz_dim = st.selectbox("Visualization Dimension:", [2, 3], index=0)
            num_viz_vectors = st.slider("Number of vectors to plot:", 1, 5, 3, key="viz_num")
            
            vectors_viz = []
            labels_viz = []
            
            for i in range(num_viz_vectors):
                st.write(f"**Vector {i+1}:**")
                vector = []
                cols = st.columns(viz_dim)
                for j in range(viz_dim):
                    with cols[j]:
                        default_val = (i+1) * (1.0 if j == 0 else 0.5)
                        val = st.number_input(
                            f"Component {j+1}",
                            value=default_val,
                            key=f"viz_{i}_{j}",
                            format="%.2f"
                        )
                        vector.append(val)
                
                label = st.text_input(f"Label for Vector {i+1}:", value=f"v{i+1}", key=f"label_{i}")
                vectors_viz.append(vector)
                labels_viz.append(label)
        
        with col2:
            st.subheader("📈 Vector Plot")
            
            if vectors_viz:
                if viz_dim == 2:
                    fig = plot_2d_vectors(vectors_viz, labels_viz)
                    st.plotly_chart(fig, use_container_width=True)
                else:  # 3D
                    # Extend 2D vectors to 3D if needed
                    vectors_3d = []
                    for v in vectors_viz:
                        if len(v) == 2:
                            vectors_3d.append(v + [0])
                        else:
                            vectors_3d.append(v)
                    
                    fig = plot_3d_vectors(vectors_3d, labels_viz)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Vector properties table
                st.subheader("📋 Vector Properties")
                props_data = []
                for i, vector in enumerate(vectors_viz):
                    magnitude = vector_magnitude(vector)
                    props_data.append({
                        'Vector': labels_viz[i],
                        'Components': str(vector),
                        'Magnitude': f"{magnitude:.6f}",
                        'Unit Vector': str([f"{x:.4f}" for x in vector_normalize(vector)])
                    })
                
                props_df = pd.DataFrame(props_data)
                st.dataframe(props_df, use_container_width=True)
    
    with tab6:
        st.header("📚 Vector Spaces Theory")
        
        st.markdown("""
        ## 🎯 What is a Vector Space?
        
        A **vector space** (or linear space) is a collection of objects called vectors, which can be added together 
        and multiplied by scalars, satisfying specific axioms.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ➕ Vector Space Axioms
            
            For vectors **u**, **v**, **w** and scalars **a**, **b**:
            
            **Closure:**
            - u + v is in the vector space
            - a·u is in the vector space
            
            **Associativity:**
            - (u + v) + w = u + (v + w)
            - a(bu) = (ab)u
            
            **Commutativity:**
            - u + v = v + u
            
            **Identity Elements:**
            - u + 0 = u (zero vector)
            - 1·u = u (scalar identity)
            
            **Inverse Elements:**
            - u + (-u) = 0
            
            **Distributivity:**
            - a(u + v) = au + av
            - (a + b)u = au + bu
            """)
            
            st.markdown("""
            ### 🔗 Key Concepts
            
            **Linear Independence:**
            - Vectors are linearly independent if no vector can be written as a linear combination of others
            - Formula: c₁v₁ + c₂v₂ + ... + cₙvₙ = 0 only when all cᵢ = 0
            
            **Span:**
            - The span of vectors is the set of all their linear combinations
            - span{v₁, v₂, ..., vₙ} = {c₁v₁ + c₂v₂ + ... + cₙvₙ}
            
            **Basis:**
            - A basis is a linearly independent set that spans the entire vector space
            - Every vector in the space can be uniquely written as a linear combination of basis vectors
            
            **Dimension:**
            - The dimension of a vector space is the number of vectors in any basis
            - All bases of a vector space have the same number of vectors
            """)
        
        with col2:
            st.markdown("""
            ### 📐 Geometric Concepts
            
            **Dot Product (Inner Product):**
            - u · v = |u||v|cos(θ)
            - Measures similarity and angle between vectors
            - u · v = 0 ⟺ vectors are orthogonal
            
            **Cross Product (3D only):**
            - u × v produces a vector perpendicular to both u and v
            - |u × v| = |u||v|sin(θ) (area of parallelogram)
            - Right-hand rule determines direction
            
            **Vector Projection:**
            - proj_u(v) = ((v·u)/|u|²) × u
            - Projects vector v onto the line of vector u
            - Used in decomposing vectors into parallel and perpendicular components
            
            **Orthogonality:**
            - Vectors are orthogonal if their dot product is zero
            - Orthogonal vectors are perpendicular
            - Orthonormal vectors are orthogonal unit vectors
            """)
            
            st.markdown("""
            ### 🔄 Gram-Schmidt Process
            
            **Purpose:** Convert a linearly independent set into an orthogonal (or orthonormal) set
            
            **Steps:**
            1. Start with first vector: u₁ = v₁
            2. For each subsequent vector vₖ:
               - Subtract projections onto previous orthogonal vectors
               - uₖ = vₖ - proj_u₁(vₖ) - proj_u₂(vₖ) - ... - proj_uₖ₋₁(vₖ)
            3. Normalize to get orthonormal set: eₖ = uₖ/|uₖ|
            
            **Applications:**
            - QR decomposition
            - Orthonormal basis construction
            - Solving least squares problems
            """)
        
        st.markdown("""
        ## 🧮 Common Vector Operations
        
        | Operation | Formula | Geometric Meaning |
        |-----------|---------|-------------------|
        | Addition | u + v = (u₁+v₁, u₂+v₂, ...) | Parallelogram rule |
        | Scalar Multiplication | ku = (ku₁, ku₂, ...) | Scaling and direction |
        | Dot Product | u · v = Σuᵢvᵢ | Projection and angle |
        | Magnitude | \|u\| = √(u · u) | Length of vector |
        | Unit Vector | û = u/\|u\| | Direction without magnitude |
        | Angle | θ = cos⁻¹((u·v)/(\|u\|\|v\|)) | Angle between vectors |
        | Cross Product | u × v = (u₂v₃-u₃v₂, u₃v₁-u₁v₃, u₁v₂-u₂v₁) | Perpendicular vector |
        """)
        
        st.markdown("""
        ## 🎲 Examples and Applications
        
        **2D Vector Space (ℝ²):**
        - Standard basis: {(1,0), (0,1)}
        - Geometric interpretation: plane
        - Applications: 2D graphics, complex numbers
        
        **3D Vector Space (ℝ³):**
        - Standard basis: {(1,0,0), (0,1,0), (0,0,1)}
        - Geometric interpretation: 3D space
        - Applications: 3D graphics, physics, engineering
        
        **Function Spaces:**
        - Polynomial space: Pₙ = {a₀ + a₁x + a₂x² + ... + aₙxⁿ}
        - Trigonometric functions: span{sin(x), cos(x)}
        - Applications: signal processing, approximation theory
        
        **Matrix Spaces:**
        - Mₘₓₙ = space of m×n matrices
        - Symmetric matrices, antisymmetric matrices
        - Applications: linear transformations, systems of equations
        """)
        
        st.markdown("""
        ## 💡 Practical Tips
        
        **Checking Linear Independence:**
        1. Form a matrix with vectors as columns
        2. Calculate the rank of the matrix
        3. If rank = number of vectors, they are linearly independent
        
        **Finding a Basis:**
        1. Start with a spanning set
        2. Remove linearly dependent vectors
        3. Use Gram-Schmidt for orthogonal basis
        
        **Solving Vector Equations:**
        1. Set up the system Ax = b
        2. Use Gaussian elimination or matrix methods
        3. Check for existence and uniqueness of solutions
        """)

if __name__ == "__main__":
    main()
