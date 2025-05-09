import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.padding import Padding
from rich.layout import Layout
from rich.box import ROUNDED

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
            raise ValueError(
                "Matriz singular ou quase singular, não é possível resolver"
            )

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


def format_float(value):
    """Formata número float para exibição sem notação científica."""
    if abs(value) < 0.0001:
        # Para valores muito pequenos, arredondamos para 0
        return "0.0000"
    else:
        # Usa formatação fixa com 4 casas decimais
        return f"{value:.4f}"


def print_system_info(system_name, A, b):
    """Imprime informações do sistema usando Rich."""
    console.rule(f"[bold blue]{system_name}[/bold blue]", style="blue")

    layout = Layout()
    layout.split_column(Layout(name="matriz_a"), Layout(name="vetor_b"))

    # Imprime Matriz A
    table_A = Table(
        title=Text("Matriz A", style="bold cyan"), box=ROUNDED, show_edge=True
    )
    num_cols_A = A.shape[1]
    for col_idx in range(num_cols_A):
        table_A.add_column(f"x{col_idx+1}", justify="center", style="green")

    for row in A:
        table_A.add_row(*[format_float(x) for x in row])

    # Imprime Vetor b
    table_b = Table(
        title=Text("Vetor b", style="bold cyan"), box=ROUNDED, show_edge=True
    )
    table_b.add_column("Valor", justify="center", style="blue")
    # Verifica se b é um escalar ou iterável
    if hasattr(b, "__iter__"):
        for val in b:
            table_b.add_row(format_float(val))
    else:
        table_b.add_row(format_float(b))

    layout["matriz_a"].update(Padding(table_A, (1, 0)))
    layout["vetor_b"].update(Padding(table_b, (1, 0, 1, 0)))

    console.print(layout)


def print_solution(solucao, A, b_orig):
    """Imprime a solução e a verificação usando Rich."""
    # Prepara a solução formatada
    formatted_solution = [format_float(x) for x in solucao]
    solution_str = "[ " + ", ".join(formatted_solution) + " ]"

    console.print(
        Panel(
            Text(f"Solução x = {solution_str}", style="bold green"),
            title="[bold]Resultado[/bold]",
            border_style="green",
            box=ROUNDED,
        )
    )

    # Verificação
    verificacao = np.dot(A, solucao)

    # Formata resultados para exibição
    verif_formatted = [format_float(x) for x in verificacao]
    b_formatted = [format_float(x) for x in b_orig]

    verif_str = "[ " + ", ".join(verif_formatted) + " ]"
    b_str = "[ " + ", ".join(b_formatted) + " ]"

    # Calcula e formata as diferenças
    diff = np.abs(verificacao - b_orig)
    diff_formatted = [format_float(x) for x in diff]
    diff_str = "[ " + ", ".join(diff_formatted) + " ]"

    # Tabela de verificação
    table = Table(title="Verificação da Solução", box=ROUNDED)
    table.add_column("Item", style="cyan")
    table.add_column("Valor", style="yellow")

    table.add_row("A·x", verif_str)
    table.add_row("b", b_str)
    table.add_row("|A·x - b|", diff_str)

    console.print(table)


# Exemplo 1: Sistema 3x3
A1 = np.array([[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]], dtype=float)
b1 = np.array([1, -2, 0], dtype=float)

print_system_info("Sistema 1 (Gauss-Jordan)", A1, b1)
try:
    solucao1 = gauss_jordan(A1, b1)
    print_solution(solucao1, A1, b1)
except ValueError as e:
    console.print(
        Panel(
            Text(str(e), style="bold red"),
            title="[bold]Erro[/bold]",
            border_style="red",
            box=ROUNDED,
        )
    )


# Descomente para executar os exemplos adicionais
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
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red", box=ROUNDED))

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
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red", box=ROUNDED))

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
    console.print(Panel(Text(str(e), style="bold red"), title="[bold]Erro[/bold]", border_style="red", box=ROUNDED))
"""
