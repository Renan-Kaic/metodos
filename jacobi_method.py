import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.padding import Padding
from rich.progress import Progress  # Para mostrar o progresso das iterações

# Inicializa o console do Rich
console = Console()


# --- Função jacobi_method (modificada para retornar também 'converges_guaranteed' e 'max_iter_reached') ---
def jacobi_method_rich(A, b, x0=None, tol=1e-6, max_iter=100, verbose=False):
    """
    Resolve o sistema de equações lineares Ax = b usando o método iterativo de Jacobi.
    Versão adaptada para feedback com Rich.

    Retorna:
    tuple: (x, iteracoes, erro_final, erro_hist, diag_dominant_warn, max_iter_reached)
    """
    n = len(b)
    A_f = A.astype(float)
    b_f = b.astype(float)

    diag_dominant_warn_lines = []
    for i in range(n):
        if np.isclose(A_f[i, i], 0.0):
            console.print(
                Panel(
                    Text(
                        f"Elemento da diagonal A[{i},{i}] é zero. O método de Jacobi não pode prosseguir.",
                        style="bold red",
                    ),
                    title="[bold red]Erro Crítico[/bold red]",
                    border_style="red",
                )
            )
            return None, 0, np.inf, [], diag_dominant_warn_lines, False

        sum_abs_off_diagonal = sum(abs(A_f[i, j]) for j in range(n) if j != i)
        if abs(A_f[i, i]) <= sum_abs_off_diagonal:
            diag_dominant_warn_lines.append(i + 1)

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = x0.astype(float).copy()

    x_novo = np.zeros(n, dtype=float)
    erro_hist = []
    erro_final = np.inf
    max_iter_reached = False

    iter_count = 0
    # Usando Rich Progress para visualização
    if verbose:
        progress_bar_context = Progress(
            "[progress.description]{task.description}",
            "Erro: [progress.percentage]{task.percentage:>3.2e}",  # Mostra o erro atual
            console=console,
        )
    else:
        # Dummy context manager if not verbose
        class DummyContext:
            def __enter__(self):
                return self

            def __exit__(self, type, value, traceback):
                pass

            def add_task(self, description, total):
                return 0  # Dummy task_id

            def update(self, task_id, advance, percentage):
                pass  # Dummy update

        progress_bar_context = DummyContext()

    with progress_bar_context as progress:
        if verbose:
            task_jacobi = progress.add_task("[cyan]Jacobi", total=max_iter)

        for i_loop in range(max_iter):
            iter_count = i_loop
            for i in range(n):
                soma = np.dot(A_f[i, :i], x[:i]) + np.dot(A_f[i, i + 1 :], x[i + 1 :])
                x_novo[i] = (b_f[i] - soma) / A_f[i, i]

            erro_final = np.linalg.norm(x_novo - x, np.inf)
            erro_hist.append(erro_final)
            x = x_novo.copy()

            if verbose:
                progress.update(
                    task_jacobi, advance=1, percentage=erro_final
                )  # Atualiza com o erro

            if erro_final < tol:
                if verbose:  # Garante que a barra chegue a 100% se convergir antes
                    progress.update(
                        task_jacobi, completed=max_iter, percentage=erro_final
                    )
                break
        else:
            max_iter_reached = True

    return (
        x,
        iter_count + 1,
        erro_final,
        erro_hist,
        diag_dominant_warn_lines,
        max_iter_reached,
    )


# --- Funções de Impressão com Rich (adaptadas/novas) ---
def print_matrix_rich(matrix, title_text, R, C):
    table = Table(title=Text(title_text, style="bold magenta"))
    for col_idx in range(C):
        table.add_column(f"Col {col_idx+1}", justify="right", style="green")
    for row_idx in range(R):
        table.add_row(*[f"{matrix[row_idx, col_idx]:.4f}" for col_idx in range(C)])
    console.print(Padding(table, (1, 0)))


def print_vector_rich(vector, title_text):
    table = Table(title=Text(title_text, style="bold magenta"))
    table.add_column("Valor", justify="right", style="cyan")
    if hasattr(vector, "__iter__"):
        for val in vector:
            table.add_row(f"{val:.4f}")
    else:
        table.add_row(f"{vector:.4f}")
    console.print(Padding(table, (0, 0, 1, 0)))


def print_iterative_method_results(
    system_name,
    A,
    b,
    solucao,
    iteracoes,
    erro_final,
    erro_hist,
    diag_warn_lines,
    max_iter_r,
    tol,
    max_iter_cfg,
):
    console.rule(f"[bold cyan]{system_name}[/bold cyan]", style="cyan")
    R_A, C_A = A.shape
    print_matrix_rich(A, "Matriz A", R_A, C_A)
    print_vector_rich(b, "Vetor b")

    if diag_warn_lines:
        warn_msg = (
            "A matriz pode não ser estritamente diagonalmente dominante nas linhas: "
            + ", ".join(map(str, diag_warn_lines))
        )
        warn_msg += ". A convergência não é garantida."
        console.print(
            Panel(
                Text(warn_msg, style="yellow"),
                title="[yellow]Aviso de Convergência[/yellow]",
                border_style="yellow",
            )
        )

    if solucao is None:
        # Erro crítico já foi impresso pela função jacobi_method_rich
        return

    panel_content = Text(f"Iterações: {iteracoes}\n", style="blue")
    panel_content.append(
        f"Erro Final (L∞): {erro_final:.2e}\n",
        style="blue" if erro_final < tol else "yellow",
    )
    panel_content.append(f"Tolerância Configurada: {tol:.1e}\n", style="dim blue")
    panel_content.append(
        f"Máximo de Iterações Configurado: {max_iter_cfg}", style="dim blue"
    )

    if max_iter_r and erro_final >= tol:
        panel_content.append(
            f"\n\nAtingiu o máximo de iterações ({max_iter_cfg}) sem convergir para a tolerância desejada.",
            style="bold orange1",
        )
        border_s = "orange1"
    elif erro_final < tol:
        border_s = "green"
    else:  # Caso não tenha atingido max_iter mas também não convergiu (improvável com o break)
        border_s = "yellow"

    console.print(
        Panel(
            panel_content,
            title="[bold]Sumário da Iteração[/bold]",
            border_style=border_s,
        )
    )

    console.print(
        Panel(
            Text(
                f"Solução x = {np.array2string(solucao, formatter={'float_kind':lambda x: '%.4f' % x})}",
                style="bold green",
            ),
            title="[bold]Resultado da Solução[/bold]",
            border_style="green",
        )
    )

    verificacao = np.dot(A, solucao)
    console.print(
        Text(
            f"Verificação A*x = {np.array2string(verificacao, formatter={'float_kind':lambda x: '%.4f' % x})}",
            style="yellow",
        )
    )
    console.print(
        Text(
            f"Original    b = {np.array2string(b, formatter={'float_kind':lambda x: '%.4f' % x})}",
            style="yellow",
        )
    )
    diff = np.abs(verificacao - b)
    console.print(
        Text(
            f"Diferença |A*x - b| = {np.array2string(diff, formatter={'float_kind':lambda x: '%.4e' % x})}",
            style="dim yellow",
        )
    )

    # Opcional: Plotar histórico de erros se tiver muitos pontos
    if len(erro_hist) > 10:
        try:
            from rich.plot import Plot, ப்  # Sim, é um caractere especial para Bar!

            # Para simplificar, não vou adicionar o plot aqui, mas seria com rich.plot.Plot
            console.print(
                Padding(
                    Text(
                        f"Histórico de erros (primeiros 5): {[f'{e:.2e}' for e in erro_hist[:5]]}",
                        style="dim",
                    ),
                    (1, 0),
                )
            )
            console.print(
                Padding(
                    Text(
                        f"Histórico de erros (últimos 5): {[f'{e:.2e}' for e in erro_hist[-5:]]}",
                        style="dim",
                    ),
                    (0, 0),
                )
            )
        except ImportError:
            console.print(
                Padding(
                    Text(
                        "Biblioteca 'matplotlib' não encontrada para plotar o histórico de erros com 'rich.plot'.",
                        style="dim yellow",
                    ),
                    (1, 0),
                )
            )


# --- Exemplos ---
# Exemplo 1: Diagonalmente dominante, deve convergir
A1 = np.array(
    [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]], dtype=float
)
b1 = np.array([6, 25, -11, 15], dtype=float)
x0_1 = np.zeros(len(b1))
TOL = 1e-8
MAX_ITER = 100

console.print("\n")
sol1, it1, err1, hist1, dw1, mi1 = jacobi_method_rich(
    A1, b1, x0_1, tol=TOL, max_iter=MAX_ITER, verbose=True
)
print_iterative_method_results(
    "Sistema 1 (Jacobi)", A1, b1, sol1, it1, err1, hist1, dw1, mi1, TOL, MAX_ITER
)

"""
# Exemplo 2: Pode não ser diagonalmente dominante, ou convergir lentamente
A2 = np.array([[1, 2, -2], [1, 1, 1], [2, 2, 1]], dtype=float)
b2 = np.array([7, 2, 5], dtype=float)  # Solução exata: x=1, y=-1, z=2
x0_2 = np.zeros(len(b2))
# Para este sistema, Jacobi não converge. A matriz de iteração T_jacobi tem raio espectral > 1.
# Vamos reduzir max_iter para ver o aviso.
console.print("\n")
sol2, it2, err2, hist2, dw2, mi2 = jacobi_method_rich(
    A2, b2, x0_2, tol=1e-5, max_iter=30, verbose=True
)
print_iterative_method_results(
    "Sistema 2 (Jacobi - Esperado Não Convergir)",
    A2,
    b2,
    sol2,
    it2,
    err2,
    hist2,
    dw2,
    mi2,
    1e-5,
    30,
)

# Exemplo 3: Matriz com zero na diagonal
A3 = np.array([[0, 2, 1], [1, 1, 1], [2, 2, 1]], dtype=float)
b3 = np.array([1, 2, 3], dtype=float)
console.print("\n")
sol3, it3, err3, hist3, dw3, mi3 = jacobi_method_rich(
    A3, b3, tol=1e-5, max_iter=50
)  # verbose=False
print_iterative_method_results(
    "Sistema 3 (Jacobi - Zero na Diagonal)",
    A3,
    b3,
    sol3,
    it3,
    err3,
    hist3,
    dw3,
    mi3,
    1e-5,
    50,
)
 """
