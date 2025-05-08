import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich import print as rprint
from rich.markdown import Markdown
from rich.syntax import Syntax
import os
import time

# Inicializar o console Rich
console = Console()

def limpar_console():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')
limpar_console()

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
            console.print(f"[yellow]   |a{i+1}{i+1}| = {diagonal} ≤ {soma_nao_diagonal} = Soma dos elementos não diagonais[/yellow]")
    
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

# Função para comparar Jacobi e Gauss-Seidel
def comparar_metodos(A, b, x0=None, tol=1e-6, max_iter=100):
    """
    Compara os métodos de Jacobi e Gauss-Seidel para o mesmo sistema.
    
    Parâmetros:
    A, b, x0, tol, max_iter: Mesmos parâmetros dos métodos individuais.
    """
    # Tenta importar o método de Jacobi
    try:
        from jacobi import jacobi_method
        console.print("[green]Método de Jacobi importado com sucesso![/green]")
        
        # Resolver com Jacobi
        with console.status("[bold blue]Executando o método de Jacobi...", spinner="dots"):
            x_j, iter_j, erro_hist_j = jacobi_method(A, b, x0, tol, max_iter)
        
        # Resolver com Gauss-Seidel
        with console.status("[bold blue]Executando o método de Gauss-Seidel...", spinner="dots"):
            x_gs, iter_gs, erro_hist_gs = gauss_seidel_method(A, b, x0, tol, max_iter)
        
        # Plotar comparação
        plt.figure(figsize=(12, 7))
        plt.semilogy(range(1, len(erro_hist_j) + 1), erro_hist_j, 'bo-', label='Jacobi')
        plt.semilogy(range(1, len(erro_hist_gs) + 1), erro_hist_gs, 'ro-', label='Gauss-Seidel')
        plt.xlabel('Número de Iterações')
        plt.ylabel('Erro (escala logarítmica)')
        plt.title('Comparação de Convergência: Jacobi vs Gauss-Seidel')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Criar tabela comparativa
        table = Table(title="Comparação dos Métodos")
        table.add_column("Método", style="cyan", no_wrap=True)
        table.add_column("Iterações", style="magenta")
        table.add_column("Erro Final", style="green")
        table.add_column("Taxa de Convergência", style="yellow")
        
        # Calcular taxas de convergência aproximadas (se houver dados suficientes)
        taxa_j = "N/A"
        if len(erro_hist_j) > 5:
            taxa_j = f"{erro_hist_j[-1]/erro_hist_j[-2]:.4f}"
            
        taxa_gs = "N/A"
        if len(erro_hist_gs) > 5:
            taxa_gs = f"{erro_hist_gs[-1]/erro_hist_gs[-2]:.4f}"
        
        table.add_row(
            "Jacobi", 
            str(iter_j), 
            f"{erro_hist_j[-1]:.2e}" if erro_hist_j else "N/A",
            taxa_j
        )
        table.add_row(
            "Gauss-Seidel", 
            str(iter_gs), 
            f"{erro_hist_gs[-1]:.2e}" if erro_hist_gs else "N/A",
            taxa_gs
        )
        
        console.print(table)
        
        return {
            'jacobi': {'solucao': x_j, 'iteracoes': iter_j, 'erro_final': erro_hist_j[-1] if erro_hist_j else float('inf')},
            'gauss_seidel': {'solucao': x_gs, 'iteracoes': iter_gs, 'erro_final': erro_hist_gs[-1] if erro_hist_gs else float('inf')}
        }
    
    except ImportError:
        console.print("[bold yellow]⚠️ Módulo Jacobi não encontrado, executando apenas Gauss-Seidel[/bold yellow]")
        x_gs, iter_gs, erro_hist_gs = gauss_seidel_method(A, b, x0, tol, max_iter)
        
        # Criar tabela para Gauss-Seidel
        table = Table(title="Resultados do Método de Gauss-Seidel")
        table.add_column("Iterações", style="magenta")
        table.add_column("Erro Final", style="green")
        table.add_row(
            str(iter_gs), 
            f"{erro_hist_gs[-1]:.2e}" if erro_hist_gs else "N/A"
        )
        console.print(table)
        
        if iter_gs < max_iter:
            plotar_convergencia(erro_hist_gs, "Convergência do Método de Gauss-Seidel")
        
        return {
            'gauss_seidel': {'solucao': x_gs, 'iteracoes': iter_gs, 'erro_final': erro_hist_gs[-1] if erro_hist_gs else float('inf')}
        }

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
                eq += f"{coef:.2f}x_{j+1}"
            else:
                if coef >= 0:
                    eq += f" + {coef:.2f}x_{j+1}"
                else:
                    eq += f" - {abs(coef):.2f}x_{j+1}"
        eq += f" = {b[i]:.2f}"
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
        table.add_row(f"x_{i+1}", f"{valor:.6f}")
    
    console.print(table)
    
    # Tabela de verificação
    table = Table(title="Verificação A*x = b")
    table.add_column("A*x", style="magenta")
    table.add_column("b", style="blue")
    table.add_column("Diferença", style="red")
    
    for i in range(len(b)):
        table.add_row(
            f"{resultado[i]:.6f}",
            f"{b[i]:.6f}",
            f"{abs(resultado[i] - b[i]):.2e}"
        )
    
    console.print(table)
    console.print(f"Norma do resíduo ||A*x - b|| = [bold]{residuo:.2e}[/bold]")
    
    if residuo < 1e-5:
        console.print("[bold green]✓ A solução é precisa![/bold green]")
    else:
        console.print("[bold yellow]⚠️ A solução tem precisão limitada.[/bold yellow]")

# Execução principal
def extrair_dados_problema(texto_problema):
    """
    Extrai os dados de um problema textual para resolver um sistema linear.
    
    Parâmetros:
    texto_problema (str): Descrição textual do problema.
    
    Retorna:
    tuple: (A, b) - Matriz de coeficientes e vetor dos termos independentes.
    """
    console.print(Panel(texto_problema, title="Problema", border_style="blue"))
    console.print("[bold yellow]Extraindo dados do problema...[/bold yellow]")
    
    # Em um caso real, aqui teria código para analisar o texto
    # Neste exemplo, vamos apenas "simular" a extração
    
    # Dados extraídos do problema (aqui estamos apenas atribuindo diretamente)
    A = np.array([
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 3]
    ], dtype=float)
    
    b = np.array([100, 200, 200, 100], dtype=float)
    
    # Mostrar o que foi extraído
    console.print("[bold green]✓ Dados extraídos com sucesso![/bold green]")
    
    table = Table(title="Dados Extraídos do Problema")
    table.add_column("Elemento", style="cyan")
    table.add_column("Valor", style="green")
    
    # Mostrar matriz A formatada
    matriz_str = ""
    for i in range(A.shape[0]):
        linha = "["
        for j in range(A.shape[1]):
            linha += f" {A[i,j]:4.1f}"
        linha += " ]"
        matriz_str += linha + "\n"
    
    # Mostrar vetor b formatado
    vetor_str = "["
    for i in range(len(b)):
        vetor_str += f" {b[i]:4.1f}"
    vetor_str += " ]"
    
    table.add_row("Matriz A", f"```\n{matriz_str}```")
    table.add_row("Vetor b", f"```\n{vetor_str}```")
    
    console.print(table)
    
    return A, b

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
    table.add_row("Erro final", f"{erro_final:.2e}")
    table.add_row("Tempo de execução", f"{tempo_execucao:.4f} segundos")
    
    console.print(table)
    
    # Tabela para a solução
    sol_table = Table(title="Solução do Sistema")
    sol_table.add_column("Variável", style="cyan")
    sol_table.add_column("Valor", style="green")
    
    for i, valor in enumerate(x):
        sol_table.add_row(f"x_{i+1}", f"{valor:.6f}")
    
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
            f"{Ax[i]:.4f}",
            f"{b[i]:.4f}",
            f"{abs(Ax[i] - b[i]):.4e}"
        )
    
    console.print(ver_table)
    
    # Resumo final
    if erro_relativo < 1e-5:
        console.print(Panel(
            f"[bold green]✓ Sistema resolvido com sucesso![/bold green]\n"
            f"Resíduo: {residuo:.2e}\n"
            f"Erro relativo: {erro_relativo:.2e}",
            title="Conclusão",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠️ Sistema resolvido com precisão limitada.[/bold yellow]\n"
            f"Resíduo: {residuo:.2e}\n"
            f"Erro relativo: {erro_relativo:.2e}",
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
    - **Erro final obtido:** {:.2e}
    - **Tempo de processamento:** {:.4f} s
    
    ### Solução Encontrada
    {}
    
    ### Verificação
    O resíduo da solução (||A*x - b||) foi {:.2e}, com erro relativo de {:.2e}.
    
    ### Conclusão
    {}
    """.format(
        iteracoes, 
        1e-6, 
        erro_final,
        tempo_execucao,
        ", ".join([f"x_{i+1} = {valor:.6f}" for i, valor in enumerate(x)]),
        residuo,
        erro_relativo,
        "A solução encontrada é precisa e confiável." if erro_relativo < 1e-5 else 
        "A solução encontrada tem precisão limitada, mas é aceitável para o problema."
    )))

if __name__ == "__main__":
    console.print(Markdown("# Método de Gauss-Seidel para Sistemas Lineares"), style="bold blue")

    # Exemplo 1: Sistema com dominância diagonal
    A1 = np.array([[10, -1, 2], 
                [-1, 11, -1], 
                [2, -1, 10]], dtype=float)
    b1 = np.array([6, 25, -11], dtype=float)

    console.rule("[bold cyan]Sistema 1: Com Dominância Diagonal[/bold cyan]")
    imprimir_sistema(A1, b1, "Sistema 1")
    
    try:
        x1, iter1, erro_hist1 = gauss_seidel_method(A1, b1)
        
        console.print(Panel(
            f"[bold green]✓ Solução encontrada após {iter1} iterações[/bold green]\n"
            f"Erro final: {erro_hist1[-1]:.2e}",
            title="Resultado", 
            border_style="green"
        ))
        
        verificar_solucao(A1, b1, x1, "Sistema 1")
        
        if iter1 < 100:  # Se convergiu
            plotar_convergencia(erro_hist1, "Convergência do Método de Gauss-Seidel - Sistema 1")
    except Exception as e:
        console.print(f"[bold red]Erro ao resolver Sistema 1: {e}[/bold red]")

    # Exemplo 2: Sistema sem dominância diagonal
    A2 = np.array([[2, 3, 1], 
                [1, -1, 4], 
                [4, 2, 1]], dtype=float)
    b2 = np.array([7, 3, 6], dtype=float)

    console.rule("[bold cyan]Sistema 2: Sem Dominância Diagonal[/bold cyan]")
    imprimir_sistema(A2, b2, "Sistema 2")
    
    try:
        x2, iter2, erro_hist2 = gauss_seidel_method(A2, b2, max_iter=50)
        
        if iter2 < 50:  # Se convergiu
            console.print(Panel(
                f"[bold green]✓ Solução encontrada após {iter2} iterações[/bold green]\n"
                f"Erro final: {erro_hist2[-1]:.2e}",
                title="Resultado", 
                border_style="green"
            ))
            
            verificar_solucao(A2, b2, x2, "Sistema 2")
            plotar_convergencia(erro_hist2, "Convergência do Método de Gauss-Seidel - Sistema 2")
        else:
            console.print(Panel(
                f"[bold red]⚠️ O método não convergiu no número máximo de {iter2} iterações.[/bold red]\n"
                f"Erro final: {erro_hist2[-1]:.2e}",
                title="Resultado", 
                border_style="red"
            ))
    except Exception as e:
        console.print(f"[bold red]Erro ao resolver Sistema 2: {e}[/bold red]")

    # Exemplo 3: Comparação com Jacobi
    console.rule("[bold cyan]Sistema 3: Comparação entre Jacobi e Gauss-Seidel[/bold cyan]")
    A3 = np.array([[5, 1, 1], 
                [1, 6, 1], 
                [1, 1, 7]], dtype=float)
    b3 = np.array([10, 15, 20], dtype=float)

    imprimir_sistema(A3, b3, "Sistema 3")
    
    try:
        resultados = comparar_metodos(A3, b3)
    except Exception as e:
        console.print(f"[bold red]Erro ao comparar métodos: {e}[/bold red]")
        
    # Exemplo 4: Problema aplicado - Distribuição de temperatura
    console.rule("[bold cyan]Exemplo Prático: Problema de Distribuição de Temperatura[/bold cyan]")
    
    problema_texto = """
    Uma barra metálica de 1 metro está submetida às seguintes condições de contorno:
    - Temperatura de 100°C na extremidade esquerda
    - Temperatura de 100°C na extremidade direita
    - Fontes de calor de 200°C nos dois pontos centrais
    
    Utilizando uma malha com 5 pontos (incluindo as extremidades), encontrar a distribuição de temperatura
    ao longo da barra no estado estacionário, utilizando o método das diferenças finitas.
    
    Este problema resulta no seguinte sistema linear Ax = b:
    - Nós internos: -x_{i-1} + 4x_i - x_{i+1} = 0 (equação da difusão de calor)
    - Condições de contorno: x_1 = 100, x_5 = 100
    - Fontes de calor: Adicionar 200 ao lado direito das equações para os nós 2 e 3
    """
    
    import time
    
    # Extrair dados do problema
    A_problema, b_problema = extrair_dados_problema(problema_texto)
    
    # Resolver o sistema
    console.print("[bold]Resolvendo o sistema com o método de Gauss-Seidel...[/bold]")
    
    start_time = time.time()
    x_problema, iter_problema, erro_hist_problema = gauss_seidel_method(A_problema, b_problema, tol=1e-8)
    end_time = time.time()
    
    tempo_execucao = end_time - start_time
    
    # Anotar resultados
    anotar_resultados(A_problema, b_problema, x_problema, iter_problema, erro_hist_problema[-1], tempo_execucao)
    
    # Plotar solução (distribuição de temperatura)
    plt.figure(figsize=(10, 6))
    x_vals = np.linspace(0, 1, len(x_problema) + 2)  # Inclui os pontos de contorno
    # Adicionar pontos de contorno (100°C nas extremidades)
    temp_vals = np.concatenate(([100], x_problema, [100]))
    
    plt.plot(x_vals, temp_vals, 'ro-', linewidth=2, markersize=10)
    plt.grid(True)
    plt.xlabel('Posição na barra (m)')
    plt.ylabel('Temperatura (°C)')
    plt.title('Distribuição de Temperatura na Barra')
    
    # Adicionar anotações nos pontos
    for i, (x, y) in enumerate(zip(x_vals, temp_vals)):
        plt.annotate(f'{y:.1f}°C', xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center')
    
    plt.show()
    
    # Plotar convergência
    plotar_convergencia(erro_hist_problema, 'Convergência para o Problema da Distribuição de Temperatura')