import matplotlib.pyplot as plt

def plot_query_times(query_times, search_function):
    """
    Plota os tempos de execução para cada query.
    
    Args:
        query_times (list): lista de tempos de execução para cada query
        search_function: função de busca utilizada (para título do gráfico)
    """
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(query_times)), query_times, marker='o')
    plt.xlabel('Query ID')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Execution Time per Query for {search_function.__name__}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    