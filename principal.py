''' DESCRIPCION DEL DATASET:
El dataset contiene datos referidos al consumo de cigarrillos


Las variables de interés son:
-consumo
-precio
-ingreso

'''

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import ttest_1samp
from scipy.stats import normaltest
from scipy.stats import linregress

def mostrar_medidas_tcentral(datos):
    print('----Medidas de tendencia central----')
    print('Media muestral:')
    print(datos.mean().round(2))
    print('Mediana: ')
    print(datos.median().round(2))
    print('Moda(s):')
    print(datos.mode())

def mostrar_medidas_dispersion(datos):
    print('----Medidas de dispersion----')
    print('Varianza:')
    print(datos.var().round(2))
    print('Desviacion tipica:')
    print(datos.std().round(2))
    print('Coeficiente de variacion:')
    cv=(datos.std(ddof=0)/datos.mean())
    print(cv.round(2))  

def mostrar_cuartiles(datos):
    print('Q1={}'.format(datos.quantile(0.25).round(2)))
    print('Q2={}'.format(datos.quantile(0.50).round(2)))
    print('Q3={}'.format(datos.quantile(0.75).round(2)))

def calcular_intervalo_varianza(datos,n):
    x1 = chi2.ppf(1-(alpha/2),n-1,loc=0,scale=1)
    x2 = chi2.ppf((alpha/2),n-1,loc=0,scale=1)
    linf = ((n-1)*datos.var())/x1
    lsup = ((n-1)*datos.var())/x2
    return linf, lsup

def test_hipotesis_varianza(datos,valor,alpha,n):
    vp = ((n-1)*datos.var())/valor
    print('CASO 1: H0: σ^2 = {} vs H1: σ^2 > {}' .format(valor,valor))
    ec_caso1 = chi2.ppf(1-alpha,n-1,loc=0,scale=1)
    print('vp={}; ec={}'.format(vp,ec_caso1))
 
    if vp > ec_caso1:
        print('Como la variable pivotal es mayor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza es mayor a {}' .format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es igual a {}' .format(valor))
    
    print('\n')
    print('CASO 2: H0: σ^2 ={} vs H1: σ^2 < {}'.format(valor,valor))
    ec_caso2 = chi2.ppf(alpha,n-1,loc=0,scale=1)
    print('vp={}; ec={}'.format(vp,ec_caso2))

    if vp < ec_caso2:
        print('Como la variable pivotal es menor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza es menor a {}'.format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es igual a {}'.format(valor))
    
    print('\n')
    print('CASO 3: H0: σ^2 = {} vs H1: σ^2 != {}'.format(valor,valor))
    ec_caso3_a = chi2.ppf((alpha/2),n-1,loc=0,scale=1)
    ec_caso3_b = chi2.ppf(1-(alpha/2),n-1,loc=0,scale=1)
    print('vp={}; ec={}'.format(vp,ec_caso3_a))
    print('vp={}; ec={}'.format(vp,ec_caso3_b))
    
    if (vp < ec_caso3_a) or (vp > ec_caso3_b):
        print('Como la variable pivotal cae en la region critica')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es {}'.format(valor))
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es igual a {}'.format(valor))

if __name__ == '__main__':

    #Lectura del archivo
    df = pd.read_csv('dataset\CigarettesB.csv',sep=',')

    #---------Estadistica descriptiva--------
    #Informacion de cada variable
    #consumo
    print('-------Variable: consumo-------')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['consumo'])
    mostrar_medidas_dispersion(df['consumo'])
    input('Presione una tecla...')
    print('\n')

    #Precio 
    print('-------Variable: precio-------')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['precio'])
    mostrar_medidas_dispersion(df['precio'])
    input('Presione una tecla...')
    print('\n')

    #Ingreso 
    print('-------Variable: ingresos-------')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['ingreso'])
    mostrar_medidas_dispersion(df['ingreso'])
    input('Presione una tecla...')
    print('\n')

    print('---CUARTILES---')
    #consumo
    print('-------Variable: consumo-------')
    mostrar_cuartiles(df['consumo'])
    input('Presione una tecla...')
    print('\n')

    #Precio 
    print('-------Variable: precio-------')
    mostrar_cuartiles(df['precio'])
    input('Presione una tecla...')
    print('\n')

    #Ingreso
    print('-------Variable: ingresos-------')
    mostrar_cuartiles(df['ingreso'])
    input('Presione una tecla...')
    print('\n')

    #Coeficiente de Asimetría y Curtosis
    print('Coeficiente de asimetria de las variables: ')
    print('consumo: {}'.format(df['consumo'].skew().round(2)))
    print('precio : {}'.format(df['precio'].skew().round(2)))
    print('ingreso : {}'.format(df['ingreso'].skew().round(2)))


    input('Presione una tecla...')
    print('\n')

    print('Curtosis:')
    print('consumo: {}'.format(df['consumo'].kurt().round(2)))
    print('precio : {}'.format(df['precio'].kurt().round(2)))
    print('ingreso: {}'.format(df['ingreso'].kurt().round(2)))


    #Graficos de histogramas y plotbox

    df['consumo'].hist(edgecolor='purple',legend=True,bins=20)
    plot.title('Histograma de consumo')
    plot.show()


    df['precio'].hist(edgecolor='purple',legend=True,bins=15)
    plot.title('Histograma de precios')
    plot.show()


    df['ingreso'].hist(edgecolor='purple',legend=True,bins=15)
    plot.title('Histograma de ingreso')
    plot.show()
     
    #---------------ESTADISTICA INFERENCIAL-----------
    alpha=0.05
    n = df['consumo'].count()#El tamaño de muestra n es el mismo para todas las variables
  
    #-----------INTERVALOS DE CONFIANZA (para una poblacion)-----------
    #Intervalo de confianza para la media
    print('\n\nINTERVALOS DE CONFIANZA PARA LA MEDIA POBLACIONAL')
 
    #consumo
    #intervalos_consumo = norm.interval(alpha,loc=df['consumo'].mean(),scale=df['consumo'].std())
    intervalos_consumo = t.interval(1-alpha,n-1,loc=df['consumo'].mean(),scale=df['consumo'].std())
    print('La media de consumo pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_consumo))
    

    #Precios
    #intervalos_precio = norm.interval(alpha,loc=df['precio'].mean(),scale=df['precio'].std())
    intervalos_precio = t.interval(1-alpha,n-1,loc=df['precio'].mean(),scale=df['precio'].std())
    print('La media de precio pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_precio))
    
    #Ingreso
    #intervalos_ingreso = norm.interval(alpha,loc=df['ingreso'].mean(),scale=df['ingreso'].std())
    intervalos_ingreso = t.interval(1-alpha,n-1,loc=df['ingreso'].mean(),scale=df['ingreso'].std())
    print('La media de ingresos pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_ingreso))


    #Intervalos de confianza para la varianza
    input('Presione una tecla...')
    print('\n\nINTERVALOS DE CONFIANZA PARA LA Varianza POBLACIONAL')

    #consumo
    consumo_linf,consumo_lsup = calcular_intervalo_varianza(df['consumo'], n)
    print('La varianza de consumo pertenece al intervalo ({},{}) con una confianza del 95%' .format(consumo_linf,consumo_lsup))

    #Precio
    precio_linf, precio_lsup = calcular_intervalo_varianza(df['precio'], n)
    print('La varianza de precio pertenece al intervalo ({},{}) con una confianza del 95%' .format(precio_linf,precio_lsup))
    
    #Ingreso
    ingreso_linf , ingreso_lsup = calcular_intervalo_varianza(df['ingreso'], n) 
    print('La varianza de ingresos pertenece al intervalo ({},{}) con una confianza del 95%' .format(ingreso_linf,ingreso_lsup))
    
    input('Presione una tecla...')
    print('\n')

    #--------TEST DE HIPOTESIS--------
    #Docima para la media poblacional con una confianza del 95%
    print('----- TEST DE HIPOTESIS -----')
    print('---MEDIA POBLACIONAL\n') 
    #consumo
    #H0: Mu=4.85 vs H1: Mu != 4.85
    print('-------Variable: consumo-------')
    print('CASO 1: H0: µ=4.85 vs H1: µ != 4.85')
    test_consumo_caso1 = ttest_1samp(df['consumo'], 4.85)
    print(test_consumo_caso1)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 4.85')
    print('\n')

    #H0: Mu=4.85 vs H1: Mu > 4.85
    print('CASO 2: H0: µ=4.85 vs H1: µ > 4.85')
    test_consumo_caso2 = ttest_1samp(df['consumo'], 4.85,alternative='greater')
    print(test_consumo_caso2)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral  4.85')
    print('\n')

    #H0: Mu=4.85 vs H1: Mu < 4.85
    print('CASO 3: H0: µ=4.85 vs H1: µ < 4.85')
    test_consumo_caso3 = ttest_1samp(df['consumo'], 4.85,alternative='less')
    print(test_consumo_caso3)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 4.85')
    
    input('Presione una tecla...')
    print('\n')
    
    #precio
    #H0: Mu=0.20 vs H1: Mu != 0.20
    print('-------Variable: precio-------')
    print('CASO 1: H0: µ=0.20 vs H1: µ !=0.20')
    test_precio_caso1 = ttest_1samp(df['precio'], 0.20)
    print(test_precio_caso1)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 0.20')
    print('\n')

    #H0: Mu=0.20 vs H1: Mu > 0.20
    print('CASO 2: H0: µ=0.20 vs H1: µ > 0.20')
    test_precio_caso2 = ttest_1samp(df['precio'], 0.20,alternative='greater')
    print(test_precio_caso2)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 0.20')
    print('\n')

    #H0: Mu=0.20 vs H1: Mu < 0.20
    print('CASO 3: H0: µ=0.20 vs H1: µ < 0.20')
    test_precio_caso3 = ttest_1samp(df['precio'], 0.20,alternative='less')
    print(test_precio_caso3)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es  0.20')
    
    input('Presione una tecla...')
    print('\n')

    #Ingreso
    #H0: Mu=4.70 vs H1: Mu != 4.70
    print('-------Variable: ingresos-------')
    print('CASO 1: H0: µ=4.70 vs H1: µ != 4.70')
    test_ingreso_caso1 = ttest_1samp(df['ingreso'], 4.70)
    print(test_ingreso_caso1)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 se rechaza la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral NO es 4.70')
    print('\n')

    #H0: Mu=4.70 vs H1: Mu > 4.70
    print('CASO 2: H0: µ=4.70 vs H1: µ > 4.70')
    test_ingreso_caso2 = ttest_1samp(df['ingreso'], 4.70,alternative='greater')
    print(test_ingreso_caso2)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 se rechaza la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es mayor a 4.70')
    print('\n')

    #H0: Mu=4.70 vs H1: Mu < 4.70
    print('CASO 3: H0: µ=4.70 vs H1: µ < 4.70')
    test_ingreso_caso3 = ttest_1samp(df['ingreso'], 4.70,alternative='less')
    print(test_ingreso_caso3)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 no hay pruebas suficientes la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 4.70')
    
    input('Presione una tecla...')
    print('\n')

    print('---VARIANZA POBLACIONAL\n')
    #consumo
    #H0: a =0.04 vs H1: a > 0.04
    print('-------Variable: consumo-------')
    test_hipotesis_varianza(df['consumo'], 0.04, alpha, n)
    input('Presione una tecla...')
    print('\n')

    #Precio
    print('-------Variable: precio-------')
    test_hipotesis_varianza(df['precio'], 0.008, alpha, n)
    input('Presione una tecla...')
    print('\n')

    print('-------Variable: ingresos-------')
    test_hipotesis_varianza(df['ingreso'], 0.020, alpha, n)
    input('Presione una tecla...')
    print('\n')

    #------ CONTRASTES NO PARAMETRICOS: Prueba de bondad de ajuste ------- #
    #Se realiza el test de hipótesis para verificar si las variables siguen una distribucion normal o no
    print('------PRUEBA DE BONDAD DE AJUSTE-----')
    print('----SE PROBARA SI LAS VARIABLES SIGUEN UNA DISTRIBUCION NORMAL O NO----')
    print('H0: "X sigue una distribucion normal" vs H1: "X NO sigue una distribucion normal".')
    print('\n')

    print('-------Variable: consumo-------')
    test_normalidad_consumo = normaltest(df['consumo'])
    print(test_normalidad_consumo)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable paquetes sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')

    print('-------Variable: precio-------')
    test_normalidad_precio = normaltest(df['precio'])
    print(test_normalidad_precio)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable precio sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')

    print('-------Variable: ingresos-------')
    test_normalidad_ingreso = normaltest(df['ingreso'])
    print(test_normalidad_ingreso)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable ingreso sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')

    #--------MODELO DE REGRESION LINEAL--------
    print('Iniciando calculo de los parametros w y b')
    print('Modelo lineal simple: y=wx+b')
    x = df['ingreso']
    y = df['precio']
    coeficientes = linregress(x,y)
    print(coeficientes)
    print('Los parametos finales son: w={},b={}'.format(coeficientes.slope.round(2),coeficientes.intercept.round(2)))
    #Funcion lambda que calcula el intervalo de confianza de la pendiente
    #La funcion se basa en una funcion t student inversa los parametros son: p: probabilidad 
    #y gl: grados de libertad
    t_inv = lambda p,gl: abs(t.ppf(p/2,gl))
    ts = t_inv(0.5,x.count()-2)
    print('Intervalo de confianza del 95% para la pendiente: {} +/- {}'.format(coeficientes.slope,ts*coeficientes.stderr))
    input('Presione una tecla...')
    plot.plot(x, y, 'o', label='datos')
    plot.plot(x,coeficientes.intercept +coeficientes.slope*x, 'g', label='regresion')
    plot.legend()
    plot.show()