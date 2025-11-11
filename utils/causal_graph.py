"""
Causal graph specification using DoWhy.
"""
import dowhy
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt


def create_ihdp_causal_graph():
    """
    Create the causal graph (DAG) for the IHDP dataset.
    
    The graph structure:
    - A (sensitive attribute x10) -> M (mediator) -> Y (outcome)
    - T (treatment) -> Y (outcome)
    - X (other covariates) -> M, Y
    
    Returns:
    --------
    graph_str : str
        Graph specification in DOT format
    """
    graph_str = """
    digraph {
        A -> M;
        A -> Y;
        T -> Y;
        M -> Y;
        X1 -> M; X1 -> Y;
        X2 -> M; X2 -> Y;
        X3 -> M; X3 -> Y;
        X4 -> M; X4 -> Y;
        X5 -> M; X5 -> Y;
        X6 -> M; X6 -> Y;
        X7 -> M; X7 -> Y;
        X8 -> M; X8 -> Y;
        X9 -> M; X9 -> Y;
        X11 -> M; X11 -> Y;
        X12 -> M; X12 -> Y;
        X13 -> M; X13 -> Y;
        X14 -> M; X14 -> Y;
        X15 -> M; X15 -> Y;
        X16 -> M; X16 -> Y;
        X17 -> M; X17 -> Y;
        X18 -> M; X18 -> Y;
        X19 -> M; X19 -> Y;
        X20 -> M; X20 -> Y;
        X21 -> M; X21 -> Y;
        X22 -> M; X22 -> Y;
        X23 -> M; X23 -> Y;
        X24 -> M; X24 -> Y;
        X25 -> M; X25 -> Y;
    }
    """
    return graph_str


def visualize_causal_graph(graph_str, save_path=None):
    """
    Visualize the causal graph.
    
    Parameters:
    -----------
    graph_str : str
        Graph specification in DOT format
    save_path : str, optional
        Path to save the visualization
    """
    try:
        import pydot
        from IPython.display import Image, display
        
        graphs = pydot.graph_from_dot_data(graph_str)
        if graphs:
            if save_path:
                graphs[0].write_png(save_path)
            print(f"Causal graph visualization {'saved' if save_path else 'generated'}")
    except ImportError:
        print("pydot not available. Skipping graph visualization.")
        print("Install with: pip install pydot")


def check_identifiability(data, treatment, outcome, graph_str):
    """
    Check if the causal effect is identifiable from the graph and data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The dataset
    treatment : str
        Name of the treatment variable
    outcome : str
        Name of the outcome variable
    graph_str : str
        Graph specification in DOT format
    
    Returns:
    --------
    identified_estimand : IdentifiedEstimand
        The identified estimand
    """
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=graph_str
    )
    
    # Identify the causal estimand
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print("Identified estimand:")
    print(identified_estimand)
    
    return identified_estimand


if __name__ == "__main__":
    # Test the causal graph creation
    graph_str = create_ihdp_causal_graph()
    print("Causal graph specification:")
    print(graph_str)
    
    # Try to visualize
    visualize_causal_graph(graph_str, save_path="causal_graph.png")

