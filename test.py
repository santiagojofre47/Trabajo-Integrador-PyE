import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import pylab 
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import ttest_1samp
from scipy.stats import chi2
from scipy.stats import normaltest
from scipy.stats import linregress


def crear(datos):
    n = datos['Zona'].count()
    caro = 0
    accesible = 0
    muy_caro = 0
    for i in range(n):
        if datos['precio'][i] < 4.55:
            accesible+=1
        elif datos['precio'][i] > 4.55 and datos['precio'][i] < 4.90:
            caro+=1
        else:
            muy_caro+=1
    print(accesible+caro+muy_caro)


def calcular_modelo(w,b,x):
    '''Retorna el valor w*x+b correspondiente al modelo lineal'''
    return w*x+b

def calcular_error(y,y_):
    '''Calcula el error cuadrático medio entre el dato original (y)
       y el dato generado por el modelo (y_)'''
    N = y.count()
    error = np.sum((y-y_)**2)/N
    return error

def gradiente_descendente(w_, b_, alpha, x, y):
    '''Algoritmo del gradiente descendente para minimizar el error
       cuadrático medio'''
    N = x.count()     # Cantidad de datos


    # Gradientes: derivadas de la función de error con respecto
    # a los parámetros "w" y "b"
    dw = -(2/N)*np.sum(x*(y-(w_*x+b_)))
    db = -(2/N)*np.sum(y-(w_*x+b_))

    # Actualizar los pesos usando la fórmula del gradiente descendente
    w = w_ - alpha*dw
    b = b_ - alpha*db

    return w, b

def calcular_intervalo_varianza(datos,n,x1,x2):
    linf = ((n-1)*datos.var())/x1
    lsup = ((n-1)*datos.var())/x2
    return linf, lsup

def test_hipotesis_varianza(datos,valor,alpha2,n):
    vp = ((n-1)*datos.var())/valor
    print('CASO 1: H0: a = {} vs H1: a > {}' .format(valor,valor))
    ec_caso1 = chi2.ppf(1-alpha2,n-1,loc=0,scale=1)
    print(vp)
    print(ec_caso1)

    if vp > ec_caso1:
        print('Como la variable pivotal es mayor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es mayor a {}' .format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es mayor a {}' .format(valor))
    
    print('\n')
    print('CASO 2: H0: a ={} vs H1: a < {}'.format(valor,valor))
    ec_caso2 = chi2.ppf(alpha2,n-1,loc=0,scale=1)

    if vp < ec_caso2:
        print('Como la variable pivotal es menor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es menor a {}'.format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es menor a {}'.format(valor))
    
    print('\n')
    print('CASO 3: H0: a = {} vs H1: a != {}'.format(valor,valor))
    ec_caso3_a = chi2.ppf((alpha2/2),n-1,loc=0,scale=1)
    ec_caso3_b = chi2.ppf(1-(alpha2/2),n-1,loc=0,scale=1)

    if (vp < ec_caso3_a) or (vp > ec_caso3_b):
        print('Como la variable pivotal cae en la region critica')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es {}'.format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es {}'.format(valor))


if __name__ == '__main__':
    df = pd.read_csv('a.csv',sep=',')
    print('µ')



    input('...')
    np.random.seed(2)
    w = np.random.randn(1)[0]
    b = np.random.randn(1)[0]

    a = 0.0004#tasa de aprendizaje
    nits = 45000#numero de iteraciones

    x = df['paquetes']
    y = df['ingreso']

    error = np.zeros((nits,1))
    for i in range(nits):
        # Actualizar valor de los pesos usando el gradiente descendente
        [w, b] = gradiente_descendente(w,b,a,x,y)
        # Calcular el valor de la predicción
        y_ = calcular_modelo(w,b,x)
        # Actualizar el valor del error
        error[i] = calcular_error(y,y_)
        # Imprimir resultados cada 1000 iteraciones
        if (i+1)%1000 == 0:
            print("Iteracion {}".format(i+1))
            print("    w: {:.1f}".format(w), " b: {:.1f}".format(b))
            print("    error: {}".format(error[i]))
            print("=======================================")
    
    plt.plot(range(nits),error)
    plt.xlabel('Iteracion')
    plt.ylabel('ECM')
    plt.show()
    y_regr = calcular_modelo(w,b,x)
    plt.scatter(x,y)
    plt.plot(x,y_regr,'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()