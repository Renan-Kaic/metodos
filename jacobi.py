import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
import time

# Configurar numpy para não usar notação científica
np.set_printoptions(precision=5, suppress=True)

# Inicializar o console Rich
console = Console()


def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Resolve um sistema de equações lineares usando o método de Jacobi.

    Parâmetros:
    A: matriz dos coeficientes (numpy array)
    b: vetor dos termos independentes (numpy array)
    x0: estimativa inicial (se None, usa vetor de zeros)
    tol: tolerância para convergência
    max_iter: número máximo de iterações

    Retorna:
    x: solução encontrada
    iterations: número de iterações realizadas
    convergence_history: histórico de convergência (norma do resíduo)
    """
    n = len(A)

    # Verificar se a matriz A é diagonal dominante
    is_diag_dominant = True
    for i in range(n):
        diagonal = abs(A[i, i])
        soma_nao_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= soma_nao_diagonal:
            is_diag_dominant = False
            console.print(
                f"[yellow]⚠️ Aviso: A matriz pode não convergir pelo método de Jacobi (linha {i+1})[/yellow]"
            )
            console.print(
                f"[yellow]   |a{i+1}{i+1}| = {diagonal:.5f} ≤ {soma_nao_diagonal:.5f} = Soma dos elementos não diagonais[/yellow]"
            )

    # Estimativa inicial (se não fornecida)
    if x0 is None:
        x0 = np.zeros(n)

    # Criar arrays para armazenar soluções
    x_old = x0.copy()
    x_new = np.zeros_like(x_old)

    # Histórico de convergência
    convergence_history = []

    # Verificar se algum elemento da diagonal é zero
    if np.any(np.diag(A) == 0):
        return (
            None,
            0,
            [],
            "Não é possível aplicar o método de Jacobi: a diagonal contém zeros.",
        )

    # Iterações do método de Jacobi com barra de status
    with console.status("[bold green]Executando o método de Jacobi...", spinner="dots"):
        for iter_count in range(max_iter):
            # Iteração de Jacobi
            for i in range(n):
                soma = 0
                for j in range(n):
                    if j != i:
                        soma += A[i, j] * x_old[j]
                x_new[i] = (b[i] - soma) / A[i, i]

            # Calcular o resíduo
            residual = np.linalg.norm(x_new - x_old, np.inf)
            convergence_history.append(residual)

            # Verificar convergência
            if residual < tol:
                return x_new, iter_count + 1, convergence_history, None

            # Atualizar x_old para a próxima iteração
            x_old = x_new.copy()

    warning = "O método atingiu o número máximo de iterações sem convergir."
    if not is_diag_dominant:
        warning += " A matriz não é diagonal dominante, o que pode explicar a falta de convergência."

    return x_new, max_iter, convergence_history, warning


def format_float(value):
    """Formata um float para evitar notação científica e limitar casas decimais"""
    return f"{value:.5f}"


def imprimir_sistema(A, b, titulo):
    """
    Formata e imprime um sistema de equações lineares.

    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes.
    b (numpy.ndarray): Vetor dos termos independentes.
    titulo (str): Título do sistema.
    """
    n = len(b)
    equacoes = []

    for i in range(n):
        eq = ""
        for j in range(n):
            coef = A[i, j]
            if j == 0:
                eq += f"{coef:.5f}x_{j+1}"
            else:
                if coef >= 0:
                    eq += f" + {coef:.5f}x_{j+1}"
                else:
                    eq += f" - {abs(coef):.5f}x_{j+1}"
        eq += f" = {b[i]:.5f}"
        equacoes.append(eq)

    md = f"## {titulo}\n\n"
    for i, eq in enumerate(equacoes):
        md += f"Equação {i+1}: {eq}\n\n"

    console.print(Markdown(md))


def verificar_solucao(A, b, x, nome_sistema):
    """
    Verifica a solução encontrada, calculando A*x e comparando com b.

    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes.
    b (numpy.ndarray): Vetor dos termos independentes.
    x (numpy.ndarray): Solução calculada.
    nome_sistema (str): Identificador do sistema para exibição.
    """
    resultado = np.dot(A, x)
    residuo = np.linalg.norm(resultado - b)

    table = Table(title=f"Verificação da Solução - {nome_sistema}")
    table.add_column("Variável", style="cyan")
    table.add_column("Valor", style="green")

    for i, valor in enumerate(x):
        table.add_row(f"x_{i+1}", f"{valor:.5f}")

    console.print(table)

    # Tabela de verificação
    table = Table(title="Verificação A*x = b")
    table.add_column("A*x", style="magenta")
    table.add_column("b", style="blue")
    table.add_column("Diferença", style="red")

    for i in range(len(b)):
        table.add_row(
            f"{resultado[i]:.5f}", f"{b[i]:.5f}", f"{abs(resultado[i] - b[i]):.5f}"
        )

    console.print(table)
    console.print(f"Norma do resíduo ||A*x - b|| = [bold]{residuo:.5f}[/bold]")

    if residuo < 1e-5:
        console.print("[bold green]✓ A solução é precisa![/bold green]")
    else:
        console.print("[bold yellow]⚠️ A solução tem precisão limitada.[/bold yellow]")


def anotar_resultados(A, b, x, iteracoes, erro_final, tempo_execucao):
    """
    Anota e apresenta os resultados da solução de um sistema linear.

    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes.
    b (numpy.ndarray): Vetor dos termos independentes.
    x (numpy.ndarray): Solução encontrada.
    iteracoes (int): Número de iterações realizadas.
    erro_final (float): Erro final obtido.
    tempo_execucao (float): Tempo de execução em segundos.
    """
    # Criar uma tabela para os resultados principais
    table = Table(title="Resultados da Solução")
    table.add_column("Parâmetro", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Número de equações", str(len(b)))
    table.add_row("Número de iterações", str(iteracoes))
    table.add_row("Erro final", f"{erro_final:.5f}")
    table.add_row("Tempo de execução", f"{tempo_execucao:.5f} segundos")

    console.print(table)

    # Tabela para a solução
    sol_table = Table(title="Solução do Sistema")
    sol_table.add_column("Variável", style="cyan")
    sol_table.add_column("Valor", style="green")

    for i, valor in enumerate(x):
        sol_table.add_row(f"x_{i+1}", f"{valor:.5f}")

    console.print(sol_table)

    # Verificar a solução
    Ax = np.dot(A, x)
    residuo = np.linalg.norm(Ax - b)
    erro_relativo = residuo / np.linalg.norm(b)

    # Tabela para verificação
    ver_table = Table(title="Verificação da Solução")
    ver_table.add_column("Equação", style="cyan")
    ver_table.add_column("A*x", style="magenta")
    ver_table.add_column("b", style="blue")
    ver_table.add_column("Erro Absoluto", style="red")

    for i in range(len(b)):
        ver_table.add_row(
            f"Equação {i+1}", f"{Ax[i]:.5f}", f"{b[i]:.5f}", f"{abs(Ax[i] - b[i]):.5f}"
        )

    console.print(ver_table)

    # Resumo final
    if erro_relativo < 1e-5:
        console.print(
            Panel(
                f"[bold green]✓ Sistema resolvido com sucesso![/bold green]\n"
                f"Resíduo: {residuo:.5f}\n"
                f"Erro relativo: {erro_relativo:.5f}",
                title="Conclusão",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]⚠️ Sistema resolvido com precisão limitada.[/bold yellow]\n"
                f"Resíduo: {residuo:.5f}\n"
                f"Erro relativo: {erro_relativo:.5f}",
                title="Conclusão",
                border_style="yellow",
            )
        )

    # Sugestão para relatório
    console.print(
        Markdown(
            """
    ## Sugestão para anotação no relatório:
    
    ### Método Utilizado
    O sistema foi resolvido utilizando o método iterativo de Jacobi.
    
    ### Parâmetros da Solução
    - **Número de iterações:** {}
    - **Critério de parada:** Erro < {}
    - **Erro final obtido:** {:.5f}
    - **Tempo de processamento:** {:.5f} s
    
    ### Solução Encontrada
    {}
    
    ### Verificação
    O resíduo da solução (||A*x - b||) foi {:.5f}, com erro relativo de {:.5f}.
    
    ### Conclusão
    {}
    """.format(
                iteracoes,
                1e-6,
                erro_final,
                tempo_execucao,
                ", ".join([f"x_{i+1} = {valor:.5f}" for i, valor in enumerate(x)]),
                residuo,
                erro_relativo,
                (
                    "A solução encontrada é precisa e confiável."
                    if erro_relativo < 1e-5
                    else "A solução encontrada tem precisão limitada, mas é aceitável para o problema."
                ),
            )
        )
    )


def plotar_convergencia(erro_hist, titulo):
    """
    Plota o gráfico de convergência do método iterativo.

    Parâmetros:
    erro_hist (list): Lista com o histórico de erros.
    titulo (str): Título do gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(erro_hist) + 1), erro_hist, "bo-")
    plt.xlabel("Número de Iterações")
    plt.ylabel("Erro (escala logarítmica)")
    plt.title(titulo)
    plt.grid(True)
    plt.show()


def main():
    console.print(
        Markdown("# Método de Jacobi para Sistemas Lineares"), style="bold blue"
    )

    try:
        # Parâmetros do método
        tol = 1e-6
        max_iter = 100


        # ADICIONAR AQUI A MATRIZ E O VETOR DO SISTEMA
        A1 = np.array([[9, -3, 2], [-1, 11, -1], [2, -1, 10]], dtype=float)
        b1 = np.array([6, 25, -11], dtype=float)


        # Exibir o sistema
        console.rule("[bold cyan]Sistema Linear [/bold cyan]")
        imprimir_sistema(A1, b1, "Sistema Linear")

        # Executar o método
        console.print("\n[bold cyan]Executando o Método de Jacobi...[/bold cyan]")

        inicio = time.time()
        x1, iteracoes1, erro_hist1, warning1 = jacobi_method(
            A1, b1, tol=tol, max_iter=max_iter
        )
        fim = time.time()

        tempo_execucao1 = fim - inicio

        # Mostrar resultados
        if iteracoes1 < max_iter:  # Convergiu
            console.print(
                Panel(
                    f"[bold green]✓ Solução encontrada após {iteracoes1} iterações[/bold green]\n"
                    f"Erro final: {erro_hist1[-1]:.5f}",
                    title="Resultado",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]⚠️ O método não convergiu após {max_iter} iterações[/bold red]\n"
                    f"Erro final: {erro_hist1[-1]:.5f}",
                    title="Resultado",
                    border_style="red",
                )
            )

        if warning1:
            console.print(f"[bold red]{warning1}[/bold red]")

        # Mostrar a solução e verificar
        verificar_solucao(A1, b1, x1, "Sistema 1")

        # Apresentar resultados detalhados
        anotar_resultados(A1, b1, x1, iteracoes1, erro_hist1[-1], tempo_execucao1)

        # Plotar convergência se pyplot estiver disponível
        # Os gráficos serão exibidos apenas se o código for executado em um ambiente com suporte gráfico
        try:
            if iteracoes1 < max_iter:
                plotar_convergencia(erro_hist1, "Convergência do Método de Jacobi")

        except Exception as e:
            console.print(
                f"[yellow]Não foi possível gerar os gráficos: {str(e)}[/yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]Erro:[/bold red] {str(e)}")


if __name__ == "__main__":
    main()
