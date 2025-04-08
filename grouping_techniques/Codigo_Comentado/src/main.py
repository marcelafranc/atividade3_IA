from iris import *
from wine import *

if __name__ == '__main__':
    escolhas = [1,2,3,4,5,6,7,8,9]

    while True:
        
        print("O que quer fazer?:")
        print("[1] Mostar Iris sem agrupamento")
        print("[2] Agrupar Iris (Hierarquico)")
        print("[3] Agrupar Iris (Particional)")
        print("[4] Mostar Wine sem agrupamento")
        print("[5] Agrupar Wine (Hierarquico)")
        print("[6] Agrupar Wine (Particional)")
        print("[7] pariplotWine")
        print("[8] pariplotIris")
        print("[9] Sair")
    
        resposta = int(input())

        if resposta not in escolhas: print("Escolha uma opcao valida")

        if resposta == 1: printIris() #iris sem agrupar
        if resposta == 2: hierarquicoIris() #iris hierarquico
        if resposta == 3: particionalIris() #iris particional
        if resposta == 4: printWine() #wine sem agrupar
        if resposta == 5: hierarquicoWine() # wine hierarquico
        if resposta == 6: particionalWine() #wine particional
        if resposta == 7: pairplotWine() #wine particional
        if resposta == 8: pairplotIris() #wine particional
        if resposta == 9: exit()
