import numpy as np
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

def gauss_elimination(A, b, show_steps=False):
    """
    Resolve um sistema de equações lineares usando o método de eliminação de Gauss.
    
    Parâmetros:
    A: matriz dos coeficientes (numpy array)
    b: vetor dos termos independentes (numpy array)
    show_steps: se True, mostra os passos intermediários do método
    
    Retorna:
    x: solução encontrada
    steps: número de operações elementares realizadas
    time_history: histórico de tempo de operações
    error_msg: mensagem de erro (se houver)
    """
    # Copiar matrizes para não modificar as originais
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(A)
    
    # Verificar se a matriz é quadrada
    if A.shape[0] != A.shape[1]:
        return None, 0, [], "A matriz de coeficientes não é quadrada."
    
    # Criar a matriz aumentada [A|b]
    Ab = np.column_stack((A, b))
    
    steps = 0
    time_history = []
    start_time = time.time()
    
    # Fase de Eliminação: Transformar a matriz A em uma matriz triangular superior
    for i in range(n-1):  # Para cada linha, exceto a última
        # Encontrar o pivô máximo na coluna atual (pivotamento parcial)
        max_row = i + np.argmax(abs(Ab[i:, i]))
        
        if max_row != i:
            # Trocar linhas
            Ab[[i, max_row]] = Ab[[max_row, i]]
            steps += 1
            if show_steps:
                console.print(f"[yellow]Passo {steps}: Trocar linhas {i+1} e {max_row+1}[/yellow]")
                mostrar_matriz_aumentada(Ab, f"Matriz após troca de linhas {i+1} e {max_row+1}")
        
        # Verificar se o pivô é zero (sistema singular)
        if abs(Ab[i, i]) < 1e-10:
            return None, steps, time_history, f"O sistema pode ser singular ou mal condicionado. Pivô na posição ({i+1},{i+1}) muito próximo de zero."
        
        # Eliminar os elementos abaixo do pivô na coluna atual
        for j in range(i+1, n):  # Para cada linha abaixo da linha i
            if abs(Ab[j, i]) > 1e-10:  # Se o elemento não for zero
                factor = Ab[j, i] / Ab[i, i]
                Ab[j] = Ab[j] - factor * Ab[i]
                steps += 1
                
                if show_steps:
                    console.print(f"[green]Passo {steps}: Subtrair {factor:.5f} vezes linha {i+1} da linha {j+1}[/green]")
                    mostrar_matriz_aumentada(Ab, f"Matriz após eliminação do elemento ({j+1},{i+1})")
        
        # Registrar o tempo após cada operação importante
        current_time = time.time() - start_time
        time_history.append(current_time)
    
    # Verificar se o último pivô é zero
    if abs(Ab[n-1, n-1]) < 1e-10:
        return None, steps, time_history, "O sistema é singular: último pivô é zero."
    
    # Fase de Substituição Reversa: Resolver o sistema triangular superior
    x = np.zeros(n)
    for i in range(n-1, -1, -1):  # Para cada linha, de baixo para cima
        # Calcular a soma dos termos conhecidos
        sum_known = 0
        for j in range(i+1, n):
            sum_known += Ab[i, j] * x[j]
        
        # Calcular o valor de x_i
        x[i] = (Ab[i, n] - sum_known) / Ab[i, i]
        steps += 1
        
        if show_steps:
            console.print(f"[cyan]Passo {steps}: Calcular x_{i+1} = {x[i]:.5f}[/cyan]")
    
    # Registrar o tempo final
    current_time = time.time() - start_time
    time_history.append(current_time)
    
    return x, steps, time_history, None

def mostrar_matriz_aumentada(Ab, titulo):
    """
    Mostra a matriz aumentada [A|b] em formato de tabela.
    
    Parâmetros:
    Ab: matriz aumentada [A|b]
    titulo: título da tabela
    """
    n = Ab.shape[0]
    m = Ab.shape[1]
    
    table = Table(title=titulo, box=box.SIMPLE)
    
    # Adicionar colunas para cada variável e para o vetor b
    for j in range(m-1):
        table.add_column(f"x_{j+1}", style="cyan")
    table.add_column("b", style="green")
    
    # Adicionar linhas
    for i in range(n):
        row_data = [f"{Ab[i,j]:.5f}" for j in range(m)]
        table.add_row(*row_data)
    
    console.print(table)

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

def verificar_solucao(A, b, x, nome_sistema="Sistema"):
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

def anotar_resultados(A, b, x, operacoes, tempo_execucao):
    """
    Anota e apresenta os resultados da solução de um sistema linear.
    
    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes.
    b (numpy.ndarray): Vetor dos termos independentes.
    x (numpy.ndarray): Solução encontrada.
    operacoes (int): Número de operações elementares realizadas.
    tempo_execucao (float): Tempo de execução em segundos.
    """
    # Criar uma tabela para os resultados principais
    table = Table(title="Resultados da Solução")
    table.add_column("Parâmetro", style="cyan")
    table.add_column("Valor", style="green")
    
    table.add_row("Número de equações", str(len(b)))
    table.add_row("Número de operações elementares", str(operacoes))
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
    erro_relativo = residuo / np.linalg.norm(b) if np.linalg.norm(b) > 0 else residuo
    
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
    O sistema foi resolvido utilizando o método direto de Eliminação de Gauss.
    
    ### Parâmetros da Solução
    - **Número de operações elementares:** {}
    - **Tempo de processamento:** {:.5f} s
    
    ### Solução Encontrada
    {}
    
    ### Verificação
    O resíduo da solução (||A*x - b||) foi {:.5f}, com erro relativo de {:.5f}.
    
    ### Conclusão
    {}
    """.format(
        operacoes, 
        tempo_execucao,
        ", ".join([f"x_{i+1} = {valor:.5f}" for i, valor in enumerate(x)]),
        residuo,
        erro_relativo,
        "A solução encontrada é exata e confiável." if erro_relativo < 1e-5 else 
        "A solução encontrada tem precisão limitada, mas é aceitável para o problema."
    )))

def plotar_desempenho(tempo_hist, titulo):
    """
    Plota o gráfico de desempenho do método direto.
    
    Parâmetros:
    tempo_hist (list): Lista com o histórico de tempos.
    titulo (str): Título do gráfico.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(tempo_hist) + 1), tempo_hist, 'ro-')
    plt.xlabel('Número de Operações')
    plt.ylabel('Tempo Acumulado (segundos)')
    plt.title(titulo)
    plt.grid(True)
    plt.show()

def main():
    console.print(Markdown("# Método de Eliminação de Gauss para Sistemas Lineares"), style="bold blue")
    
    try:
        # Sistema 1: Sistema de exemplo
        A1 = np.array([
            [9, 3, 2], 
            [-1, 11, -1], 
            [2, -1, 10]
        ], dtype=float)
        b1 = np.array([6, 25, -11], dtype=float)
        
        # Exibir o sistema
        console.rule("[bold cyan]Sistema Linear [/bold cyan]")
        imprimir_sistema(A1, b1, "Sistema Linear")
        
        # Definir se deseja mostrar passos intermediários
        mostrar_passos = True  # Pode ser alterado para False se não quiser ver todos os passos
        
        # Executar o método
        console.print("\n[bold cyan]Executando o Método de Eliminação de Gauss...[/bold cyan]")
        
        inicio = time.time()
        x1, operacoes1, tempo_hist1, erro1 = gauss_elimination(A1, b1, show_steps=mostrar_passos)
        fim = time.time()
        
        tempo_execucao1 = fim - inicio
        
        # Mostrar resultados
        if x1 is not None:  # Solução encontrada
            console.print(Panel(
                f"[bold green]✓ Solução encontrada com {operacoes1} operações elementares[/bold green]",
                title="Resultado", 
                border_style="green"
            ))
        else:
            console.print(Panel(
                f"[bold red]⚠️ Não foi possível encontrar uma solução[/bold red]\n"
                f"Erro: {erro1}",
                title="Resultado", 
                border_style="red"
            ))
            return
        
        # Mostrar a solução e verificar
        verificar_solucao(A1, b1, x1, "Sistema 1")
        
        # Apresentar resultados detalhados
        anotar_resultados(A1, b1, x1, operacoes1, tempo_execucao1)
        
        # Plotar desempenho se pyplot estiver disponível
        try:
            if len(tempo_hist1) > 0:
                plotar_desempenho(tempo_hist1, "Desempenho do Método de Eliminação de Gauss")
        except Exception as e:
            console.print(f"[yellow]Não foi possível gerar os gráficos: {str(e)}[/yellow]")
        
        # Sistema 2: Sistema alternativo
        console.print("\n[bold]Exemplo de Sistema Linear Alternativo:[/bold]")
        A2 = np.array([
            [4, 2, 1], 
            [3, 8, 5], 
            [2, 7, 6]
        ], dtype=float)
        b2 = np.array([11, 15, 8], dtype=float)
        
        # Exibir o sistema
        console.rule("[bold cyan]Sistema Linear Alternativo [/bold cyan]")
        imprimir_sistema(A2, b2, "Sistema Linear Alternativo")
        
        # Executar o método para o segundo sistema
        console.print("\n[bold cyan]Executando o Método de Eliminação de Gauss para o sistema alternativo...[/bold cyan]")
        
        inicio = time.time()
        x2, operacoes2, tempo_hist2, erro2 = gauss_elimination(A2, b2, show_steps=mostrar_passos)
        fim = time.time()
        
        tempo_execucao2 = fim - inicio
        
        # Mostrar resultados do sistema alternativo
        if x2 is not None:
            console.print(Panel(
                f"[bold green]✓ Solução encontrada com {operacoes2} operações elementares[/bold green]",
                title="Resultado", 
                border_style="green"
            ))
            
            # Mostrar a solução e verificar
            verificar_solucao(A2, b2, x2, "Sistema 2")
            
            # Apresentar resultados detalhados
            anotar_resultados(A2, b2, x2, operacoes2, tempo_execucao2)
            
            # Plotar desempenho se pyplot estiver disponível
            try:
                if len(tempo_hist2) > 0:
                    plotar_desempenho(tempo_hist2, "Desempenho do Método de Eliminação de Gauss - Sistema 2")
            except Exception as e:
                console.print(f"[yellow]Não foi possível gerar os gráficos: {str(e)}[/yellow]")
        else:
            console.print(Panel(
                f"[bold red]⚠️ Não foi possível encontrar uma solução[/bold red]\n"
                f"Erro: {erro2}",
                title="Resultado", 
                border_style="red"
            ))
       
    except Exception as e:
        console.print(f"[bold red]Erro:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()