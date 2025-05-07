import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.padding import Padding

# Inicializa o console do Rich
console = Console()

def gauss_jordan(A, b):
    """
    Resolve o sistema de equações lineares Ax = b usando o método de Gauss-Jordan.

    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes do sistema.
    b (numpy.ndarray): Vetor dos termos independentes.

    Retorna:
    numpy.ndarray: Vetor solução x.
    """
    # Cria uma cópia da matriz aumentada [A|b]
    n = len(b)
    # Certifica-se de que b é um vetor coluna para np.column_stack
    if b.ndim == 1:
        b_col = b.reshape(-1, 1)
    else:
        b_col = b
    # Garante que são float para as operações
    Ab = np.column_stack((A.astype(float), b_col.astype(float)))

    # Eliminação Gauss-Jordan
    for i in range(n):
        # Pivoteamento parcial
        max_row = i + np.argmax(abs(Ab[i:, i]))
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]  # Troca de linhas

        # Verificar se a matriz é singular
        if abs(Ab[i, i]) < 1e-10:
            raise ValueError("Matriz singular ou quase singular, não é possível resolver")

        # Normalizar a linha do pivô
        pivot = Ab[i, i]
        Ab[i, :] = Ab[i, :] / pivot

        # Eliminação dos elementos acima e abaixo do pivô
        for j in range(n):
            if j != i:
                factor = Ab[j, i]
                Ab[j, :] -= factor * Ab[i, :]

    # Extrair a solução
    x = Ab[:, n]

    return x

def print_system_info(system_name, A, b):
    """Imprime informações do sistema usando Rich."""
    console.rule(f"[bold cyan]{system_name}[/bold cyan]", style="cyan")

    # Imprime Matriz A
    table_A = Table(title=Text("Matriz A", style="bold magenta"))
    num_cols_A = A.shape[1]
    for col_idx in range(num_cols_A):
        table_A.add_column(f"Col {col_idx+1}", justify="right", style="green")

    for row in A:
        table_A.add_row(*[f"{x:.4f}" for x in row]) # Formatado para 4 casas decimais
    console.print(Padding(table_A, (1, 0)))

    # Imprime Vetor b
    table_b = Table(title=Text("Vetor b", style="bold magenta"))
    table_b.add_column("Valor", justify="right", style="blue")
    # Verifica se b é um escalar ou iterável
    if hasattr(b, '__iter__'):
        for val in b:
            table_b.add_row(f"{val:.4f}") # Formatado para 4 casas decimais
    else:
        table_b.add_row(f"{b:.4f}")
    console.print(Padding(table_b, (0, 0, 1, 0)))


def print_solution(solucao, A, b_orig):
    """Imprime a solução e a verificação usando Rich."""
    console.print(Panel(Text(f"Solução x = {np.array2string(solucao, formatter={'float_kind':lambda x: '%.4f' % x})}", style="bold green"),
                        title="[bold]Resultado[/bold]", border_style="green"))

    # Verificação
    verificacao = np.dot(A, solucao)
    console.print(Text(f"Verificação A*x = {np.array2string(verificacao, formatter={'float_kind':lambda x: '%.4f' % x})}", style="yellow"))
    console.print(Text(f"Original    b = {np.array2string(b_orig, formatter={'float_kind':lambda x: '%.4f' % x})}", style="yellow"))
    diff = np.abs(verificacao - b_orig)
    console.print(Text(f"Diferença |A*x - b| = {np.array2string(diff, formatter={'float_kind':lambda x: '%.4e' % x})}", style="dim yellow"))




A1 = np.array([[3, 2, -1],
               [2, -2, 4],
               [-1, 0.5, -1]], dtype=float)
b1 = np.array([1, -2, 0], dtype=float)

print_system_info("Sistema 1 (Gauss-Jordan)", A1, b1)
try:
    solucao1 = gauss_jordan(A1, b1)
    print_solution(solucao1, A1, b1)
except ValueError as e:
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red"))

""" 
# Exemplo 2: Sistema 4x4
A2 = np.array([[2, 1, -1, 3],
               [4, -2, 1, 2],
               [1, 5, -3, -2],
               [3, 1, 2, -4]], dtype=float)
b2 = np.array([5, 3, -4, 1], dtype=float)

print_system_info("Sistema 2 (Gauss-Jordan)", A2, b2)
try:
    solucao2 = gauss_jordan(A2, b2)
    print_solution(solucao2, A2, b2)
except ValueError as e:
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red"))

# Exemplo 3: Sistema mal condicionado
A3 = np.array([[1, 1, 1],
               [1, 1.001, 1],
               [1, 1, 1.001]], dtype=float)
b3 = np.array([3, 3.001, 3.001], dtype=float)

print_system_info("Sistema 3 (mal condicionado, Gauss-Jordan)", A3, b3)
try:
    solucao3 = gauss_jordan(A3, b3)
    print_solution(solucao3, A3, b3)
except ValueError as e:
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red"))

# Exemplo 4: Matriz singular (para testar o erro)
A4 = np.array([[1, 1, 0],
               [2, 2, 0],
               [0, 0, 1]], dtype=float)
b4 = np.array([2, 4, 3], dtype=float) # sistema com infinitas solucoes se b4[1]=2*b4[0]
                                       # ou inconsistente, como está agora (2,4,3)
                                       # o pivoteamento vai tornar o pivo 0 na segunda linha

print_system_info("Sistema 4 (singular, Gauss-Jordan)", A4, b4)
try:
    solucao4 = gauss_jordan(A4, b4)
    print_solution(solucao4, A4, b4)
except ValueError as e:
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red"))
    
    """