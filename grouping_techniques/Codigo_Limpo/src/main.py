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
        print("[7] Pairplot Wine")
        print("[8] Pairplot Iris")
        print("[9] Sair")
    
        resposta = int(input())

        if resposta not in escolhas: print("Escolha uma opcao valida")

        if resposta == 1: printIris()
        if resposta == 2: hierarquicoIris()
        if resposta == 3: particionalIris()
        if resposta == 4: printWine()
        if resposta == 5: hierarquicoWine()
        if resposta == 6: particionalWine()
        if resposta == 7: pairplotWine()
        if resposta == 8: pairplotIris()
        if resposta == 9: exit()
