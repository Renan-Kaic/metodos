import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
import time

# Inicializar o console Rich
console = Console()

def gauss_seidel_method(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Resolve o sistema de equações lineares Ax = b usando o método iterativo de Gauss-Seidel.
    
    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes do sistema.
    b (numpy.ndarray): Vetor dos termos independentes.
    x0 (numpy.ndarray, opcional): Aproximação inicial para a solução. Se None, usa vetor de zeros.
    tol (float, opcional): Tolerância para critério de parada. Padrão é 1e-6.
    max_iter (int, opcional): Número máximo de iterações. Padrão é 100.
    
    Retorna:
    tuple: (x, iteracoes, erro_hist) - Solução aproximada, número de iterações e histórico de erros.
    """
    n = len(b)
    
    # Verificar critério de convergência (dominância diagonal)
    for i in range(n):
        diagonal = abs(A[i, i])
        soma_nao_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= soma_nao_diagonal:
            console.print(f"[yellow]⚠️ Aviso: A matriz pode não convergir pelo método de Gauss-Seidel (linha {i+1})[/yellow]")
            console.print(f"[yellow]   |a{i+1}{i+1}| = {diagonal:.5f} ≤ {soma_nao_diagonal:.5f} = Soma dos elementos não diagonais[/yellow]")
    
    # Inicialização
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    erro_hist = []
    
    # Iterações do método de Gauss-Seidel com barra de progresso
    with console.status("[bold green]Executando o método de Gauss-Seidel...", spinner="dots"):
        for iter_count in range(max_iter):
            x_velho = x.copy()
            
            # Atualiza cada componente de x usando numpy para maior eficiência
            for i in range(n):
                # Usa valores já atualizados para índices anteriores a i
                soma1 = np.dot(A[i, :i], x[:i]) if i > 0 else 0
                # Usa valores antigos para índices posteriores a i
                soma2 = np.dot(A[i, i+1:], x_velho[i+1:]) if i < n-1 else 0
                x[i] = (b[i] - soma1 - soma2) / A[i, i]
            
            # Calcula erro
            erro = np.linalg.norm(x - x_velho, np.inf)
            erro_hist.append(erro)
            
            # Verifica convergência
            if erro < tol:
                return x, iter_count + 1, erro_hist
    
    console.print(f"[bold red]⚠️ Alerta: Método de Gauss-Seidel não convergiu após {max_iter} iterações.[/bold red]")
    return x, max_iter, erro_hist

def plotar_convergencia(erro_hist, titulo):
    """
    Plota o gráfico de convergência do método iterativo.
    
    Parâmetros:
    erro_hist (list): Lista com o histórico de erros.
    titulo (str): Título do gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, len(erro_hist) + 1), erro_hist, 'ro-')
    plt.xlabel('Número de Iterações')
    plt.ylabel('Erro (escala logarítmica)')
    plt.title(titulo)
    plt.grid(True)
    plt.show()

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
            f"{resultado[i]:.5f}",
            f"{b[i]:.5f}",
            f"{abs(resultado[i] - b[i]):.5f}"
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
            f"Equação {i+1}",
            f"{Ax[i]:.5f}",
            f"{b[i]:.5f}",
            f"{abs(Ax[i] - b[i]):.5f}"
        )
    
    console.print(ver_table)
    
    # Resumo final
    if erro_relativo < 1e-5:
        console.print(Panel(
            f"[bold green]✓ Sistema resolvido com sucesso![/bold green]\n"
            f"Resíduo: {residuo:.5f}\n"
            f"Erro relativo: {erro_relativo:.5f}",
            title="Conclusão",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️ Sistema resolvido com precisão limitada.[/bold yellow]\n"
            f"Resíduo: {residuo:.5f}\n"
            f"Erro relativo: {erro_relativo:.5f}",
            title="Conclusão",
            border_style="yellow"
        ))
    
    # Sugestão para relatório
    console.print(Markdown("""
    ## Sugestão para anotação no relatório:
    
    ### Método Utilizado
    O sistema foi resolvido utilizando o método iterativo de Gauss-Seidel.
    
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
        "A solução encontrada é precisa e confiável." if erro_relativo < 1e-5 else 
        "A solução encontrada tem precisão limitada, mas é aceitável para o problema."
    )))

if __name__ == "__main__":
    # Configura numpy para usar 5 casas decimais e não usar notação científica
    np.set_printoptions(precision=5, suppress=True)
    
    console.print(Markdown("# Método de Gauss-Seidel para Sistemas Lineares"), style="bold blue")
    
    # SISTEMA PREDEFINIDO - Modifique aqui para resolver diferentes sistemas
    # ======================================================================
    # Sistema 1: Com dominância diagonal
    A = np.array([
        [9, -3, 2], 
        [-1, 11, -1], 
        [2, -1, 10]
    ], dtype=float)
    b = np.array([6, 25, -11], dtype=float)
    
    # Parâmetros do método
    tol = 1e-6
    max_iter = 100
    
    # Exibir o sistema
    console.rule("[bold cyan]Sistema Linear a ser Resolvido[/bold cyan]")
    imprimir_sistema(A, b, "Sistema Linear")
    
    # Executar o método
    console.print("\n[bold cyan]Executando o Método de Gauss-Seidel...[/bold cyan]")
    
    inicio = time.time()
    x, iteracoes, erro_hist = gauss_seidel_method(A, b, tol=tol, max_iter=max_iter)
    fim = time.time()
    
    tempo_execucao = fim - inicio
    
    # Mostrar resultados
    if iteracoes < max_iter:  # Convergiu
        console.print(Panel(
            f"[bold green]✓ Solução encontrada após {iteracoes} iterações[/bold green]\n"
            f"Erro final: {erro_hist[-1]:.5f}",
            title="Resultado", 
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold red]⚠️ O método não convergiu após {max_iter} iterações[/bold red]\n"
            f"Erro final: {erro_hist[-1]:.5f}",
            title="Resultado", 
            border_style="red"
        ))
    
    # Mostrar a solução e verificar
    verificar_solucao(A, b, x, "Sistema")
    
    # Apresentar resultados detalhados
    anotar_resultados(A, b, x, iteracoes, erro_hist[-1], tempo_execucao)
    
    # Plotar convergência se convergiu
    if iteracoes < max_iter:
        plotar_convergencia(erro_hist, "Convergência do Método de Gauss-Seidel")