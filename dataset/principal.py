''' DESCRIPCION DEL DATASET:
El dataset contiene datos referidos al consumo de cigarrillos


Las variables de interés son:
-paquetes
-precio
-ingreso

'''

import pandas as pd
import math
import matplotlib.pyplot as plot
import numpy as np
#from statsmodels.graphics.gofplots import qqplot
from scipy.stats import norm
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import ttest_1samp
from scipy.stats import normaltest

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

def mostrar_medidas_tcentral(datos):
    print('----Medidas de tendencia central----')
    print('Media muestral:')
    print(datos.mean())
    print('Mediana: ')
    print(datos.median())

def mostrar_medidas_dispersion(datos):
    print('----Medidas de dispersion----')
    print('Varianza:')
    print(datos.var())
    print('Desviacion tipica:')
    print(datos.std())
    print('Coeficiente de variacion:')
    cv=(datos.std(ddof=0)/datos.mean())
    print(cv)  

def mostrar_cuartiles(datos):
    print('Q1={}'.format(datos.quantile(0.25)))
    print('Q2={}'.format(datos.quantile(0.50)))
    print('Q3={}'.format(datos.quantile(0.75)))

def calcular_intervalo_varianza(datos,n,x1,x2):
    linf = ((n-1)*datos.var())/x1
    lsup = ((n-1)*datos.var())/x2
    return linf, lsup



if __name__ == '__main__':

    #Lectura del archivo
    df = pd.read_csv('dataset\CigarettesB.csv',sep=',')

    #---------Estadistica descriptiva--------
    #Informacion de cada variable
    #Paquetes
    print('--paquetes')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['paquetes'])
    mostrar_medidas_dispersion(df['paquetes'])
    input('Presione una tecla...')
    print('\n')

    #Precio 
    print('--precio')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['precio'])
    mostrar_medidas_dispersion(df['precio'])
    input('Presione una tecla...')
    print('\n')

    #Ingreso 
    print('--ingreso')
    print('Tipo de variable: cuantitativa continua')
    mostrar_medidas_tcentral(df['ingreso'])
    mostrar_medidas_dispersion(df['ingreso'])
    input('Presione una tecla...')
    print('\n')

    print('---CUARTILES---')
    #Paquetes
    print('--paquetes')
    mostrar_cuartiles(df['paquetes'])
    input('Presione una tecla...')
    print('\n')

    #Precio 
    print('--precio')
    mostrar_cuartiles(df['precio'])
    input('Presione una tecla...')
    print('\n')

    #Ingreso
    print('--ingreso')
    mostrar_cuartiles(df['ingreso'])
    input('Presione una tecla...')
    print('\n')

    #Coeficiente de Asimetría y Curtosis
    print('Coeficiente de asimetria de las variables: ')
    print('paquetes: {}'.format(df['paquetes'].skew()))
    print('precio : {}'.format(df['precio'].skew()))
    print('ingreso : {}'.format(df['ingreso'].skew()))


    input('Presione una tecla...')
    print('\n')

    print('Curtosis:')
    print('Coeficiente de asimetria de las variables: ')
    print('paquetes: {}'.format(df['paquetes'].kurt()))
    print('precio : {}'.format(df['precio'].kurt()))
    print('ingreso: {}'.format(df['ingreso'].kurt()))


    #Graficos de histogramas y plotbox
    df['paquetes'].hist(rwidth=0.85)
    plot.show()
    df['precio'].hist(rwidth=0.85)
    plot.show()
    df['ingreso'].hist()
    plot.show()

    df['paquetes'].plot(kind='box')
    plot.show()
    df['precio'].plot(kind='box')
    plot.show()
    df['ingreso'].plot(kind='box')
    plot.show()
     
    #---------------ESTADISTICA INFERENCIAL-----------
    alpha=0.95
    alpha2 = 1-alpha
    n = df['paquetes'].count()#El tamaño de muestra n es el mismo para todas las variables
  
    #-----------INTERVALOS DE CONFIANZA (para una poblacion)-----------
    #Intervalo de confianza para la media
    print('\n\nINTERVALOS DE CONFIANZA PARA LA MEDIA POBLACIONAL')
 
    #Paquetes
    #intervalos_paquetes = norm.interval(alpha,loc=df['paquetes'].mean(),scale=df['paquetes'].std())
    intervalos_paquetes = t.interval(alpha,n-1,loc=df['paquetes'].mean(),scale=df['paquetes'].std())
    print('La media de paquetes pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_paquetes))
    

    #Precios
    #intervalos_precio = norm.interval(alpha,loc=df['precio'].mean(),scale=df['precio'].std())
    intervalos_precio = t.interval(alpha,n-1,loc=df['precio'].mean(),scale=df['precio'].std())
    print('La media de precio pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_precio))
    
    #Ingreso
    #intervalos_ingreso = norm.interval(alpha,loc=df['ingreso'].mean(),scale=df['ingreso'].std())
    intervalos_ingreso = t.interval(alpha,n-1,loc=df['ingreso'].mean(),scale=df['ingreso'].std())
    print('La media de ingresos pertenece al intervalo: {} con una confianza del 95%'.format(intervalos_ingreso))


    #Intervalos de confianza para la varianza
    input('Presione una tecla...')
    print('\n\nINTERVALOS DE CONFIANZA PARA LA Varianza POBLACIONAL')
    x1 = chi2.ppf(1-(alpha/2),n-1,loc=0,scale=1)
    x2 = chi2.ppf((alpha/2),n-1,loc=0,scale=1)

    #Paquetes
    paquetes_linf,paquetes_lsup = calcular_intervalo_varianza(df['paquetes'], n, x1, x2)
    print('La varianza de paquetes pertenece al intervalo ({},{}) con una confianza del 95%' .format(paquetes_linf,paquetes_lsup))

    #Precio
    precio_linf, precio_lsup = calcular_intervalo_varianza(df['precio'], n, x1, x2)
    print('La varianza de precio pertenece al intervalo ({},{}) con una confianza del 95%' .format(precio_linf,precio_lsup))
    
    #Ingreso
    ingreso_linf , ingreso_lsup = calcular_intervalo_varianza(df['ingreso'], n, x1, x2) 
    print('La varianza de ingresos pertenece al intervalo ({},{}) con una confianza del 95%' .format(paquetes_linf,paquetes_lsup))
    
    input('Presione una tecla...')
    print('\n')
    #--------TEST DE HIPOTESIS--------
    #Docima para la media poblacional con una confianza del 95%
    print('----- TEST DE HIPOTESIS -----')
    print('---MEDIA POBLACIONAL\n') 
    #Paquetes
    #H0: Mu=4.85 vs H1: Mu != 4.85
    print('-------Variable: paquetes-------')
    print('CASO 1: H0: Mu=4.85 vs H1: Mu != 4.85')
    test_paquetes_caso1 = ttest_1samp(df['paquetes'], 4.85)
    print(test_paquetes_caso1)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 4.85')
    print('\n')

    #H0: Mu=4.85 vs H1: Mu > 4.85
    print('CASO 2: H0: Mu=4.85 vs H1: Mu > 4.85')
    test_paquetes_caso2 = ttest_1samp(df['paquetes'], 4.85,alternative='greater')
    print(test_paquetes_caso2)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es mayor a 4.85')
    print('\n')

    #H0: Mu=4.85 vs H1: Mu < 4.85
    print('CASO 3: H0: Mu=4.85 vs H1: Mu < 4.85')
    test_paquetes_caso3 = ttest_1samp(df['paquetes'], 4.85,alternative='less')
    print(test_paquetes_caso3)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es menor a 4.85')
    
    input('Presione una tecla...')
    print('\n')
    
    #precio
    #H0: Mu=0.20 vs H1: Mu != 0.20
    print('-------Variable: precio-------')
    print('CASO 1: H0: Mu=0.20 vs H1: Mu !=0.20')
    test_precio_caso1 = ttest_1samp(df['precio'], 0.20)
    print(test_precio_caso1)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es 0.20')
    print('\n')

    #H0: Mu=0.20 vs H1: Mu > 0.20
    print('CASO 2: H0: Mu=0.20 vs H1: Mu > 0.20')
    test_precio_caso2 = ttest_1samp(df['precio'], 0.20,alternative='greater')
    print(test_precio_caso2)
    print('Luego, como el p-valor obtenido es mayor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es mayor a 0.20')
    print('\n')

    #H0: Mu=0.20 vs H1: Mu < 0.20
    print('CASO 3: H0: Mu=0.20 vs H1: Mu < 0.20')
    test_precio_caso3 = ttest_1samp(df['precio'], 0.20,alternative='less')
    print(test_precio_caso3)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 no hay pruebas suficientes para rechazar la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es menor a 0.20')
    
    input('Presione una tecla...')
    print('\n')

    #Ingreso
    #H0: Mu=4.70 vs H1: Mu != 4.70
    print('-------Variable: ingresos-------')
    print('CASO 1: H0: Mu=4.70 vs H1: Mu != 4.70')
    test_ingreso_caso1 = ttest_1samp(df['ingreso'], 4.70)
    print(test_ingreso_caso1)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 se rechaza la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral NO es 4.70')
    print('\n')

    #H0: Mu=4.70 vs H1: Mu > 4.70
    print('CASO 2: H0: Mu=4.70 vs H1: Mu > 4.70')
    test_ingreso_caso2 = ttest_1samp(df['ingreso'], 4.70,alternative='greater')
    print(test_ingreso_caso2)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 se rechaza la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral NO es mayor a 4.70')
    print('\n')

    #H0: Mu=4.70 vs H1: Mu < 4.70
    print('CASO 3: H0: Mu=4.70 vs H1: Mu < 4.70')
    test_ingreso_caso3 = ttest_1samp(df['ingreso'], 4.70,alternative='less')
    print(test_ingreso_caso3)
    print('Luego, como el p-valor obtenido es menor a alpha=0.05 no hay pruebas suficientes la hipotesis nula.')
    print('Con una confianza del 95% se puede afirmar que la media muestral es menor a 4.70')
    
    input('Presione una tecla...')
    print('\n')

    print('---VARIANZA POBLACIONAL\n')
    #Paquetes
    #H0: a =0.04 vs H1: a > 0.04
    print('Variable: paquetes')
    vp_paquetes = ((n-1)*df['paquetes'].var())/0.04
  
    print('CASO 1: H0: a =0.04 vs H1: a > 0.04')
    ec_paquetes_caso1 = chi2.ppf(1-alpha2,n-1,loc=0,scale=1)
    print(vp_paquetes)
    print(ec_paquetes_caso1)

    if vp_paquetes > ec_paquetes_caso1:
        print('Como la variable pivotal es mayor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es mayor a 0.04')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es mayor a 0.04')
    
    print('\n')
    print('CASO 2: H0: a =0.04 vs H1: a < 0.04')
    ec_paquetes_caso2 = chi2.ppf(alpha2,n-1,loc=0,scale=1)

    if vp_paquetes < ec_paquetes_caso2:
        print('Como la variable pivotal es menor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es menor a 0.04')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es menor a 0.04')
    
    print('\n')
    print('CASO 3: H0: a = 0.04 vs H1: a !=0.04')
    ec_paquetes_caso3_a = chi2.ppf((alpha2/2),n-1,loc=0,scale=1)
    ec_paquetes_caso3_b = chi2.ppf(1-(alpha2/2),n-1,loc=0,scale=1)

    if (vp_paquetes< ec_paquetes_caso3_a) or (vp_paquetes> ec_paquetes_caso3_b):
        print('Como la variable pivotal cae en la region critica')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es 0.04')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es 0.04')

    input('Presione una tecla...')
    print('\n')

    #Precio
    print('Variable: precio')
    vp_precio = ((n-1)*df['precio'].var())/0.008

    print('CASO 1: H0: a = 0.008 vs H1: a > 0.008')
    ec_precio_caso1 = chi2.ppf(1-alpha2,n-1,loc=0,scale=1)
   
    if vp_precio > ec_precio_caso1:
        print('Como la variable pivotal es mayor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es mayor a 0.008')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es mayor a 0.008')
    
    print('\n')

    print('CASO 2: H0: a = 0.008 vs H1: a < 0.008')
    ec_precio_caso2 = chi2.ppf(alpha2,n-1,loc=0,scale=1)
    
    if vp_precio < ec_precio_caso2:
        print('Como la variable pivotal es menor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es menor a 0.008')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es menor a 0.008')
    print('\n')

    print('CASO 3: H0: a = 0.008 vs H1: a != 0.008')
    ec_precio_caso3_a = chi2.ppf((alpha2/2),n-1,loc=0,scale=1)
    ec_precio_caso3_b = chi2.ppf(1-(alpha2/2),n-1,loc=0,scale=1)

    if (vp_precio < ec_precio_caso3_a) or (vp_precio> ec_precio_caso3_b):
        print('Como la variable pivotal cae en la region critica')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es 0.008')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es 0.008')
    
    input('Presione una tecla...')
    print('\n')

    print('Variable: ingreso')
    vp_ingreso = ((n-1)*df['ingreso'].var())/0.020

    print('CASO 1: H0: a = 0.020 vs H1: a > 0.020')
    ec_ingreso_caso1 = chi2.ppf(1-alpha2,n-1,loc=0,scale=1)

    if vp_ingreso > ec_ingreso_caso1:
        print('Como la variable pivotal es mayor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es mayor a 0.020')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es mayor a 0.020')
    
    print('\n')
    print('CASO 2: H0: a = 0.020 vs H1: a < 0.020')
    ec_ingreso_caso2 = chi2.ppf(alpha2,n-1,loc=0,scale=1)
    
    if vp_ingreso < ec_ingreso_caso2:
        print('Como la variable pivotal es menor al estadístico de comparacion')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es menor a 0.020')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es menor a 0.020')
    
    print('\n')
    print('CASO 3: H0: a = 0.020 vs H1: a != 0.020')
    ec_ingreso_caso3_a = chi2.ppf((alpha2/2),n-1,loc=0,scale=1)
    ec_ingreso_caso3_b = chi2.ppf(1-(alpha2/2),n-1,loc=0,scale=1)

    if (vp_ingreso < ec_ingreso_caso3_a) or (vp_ingreso> ec_ingreso_caso3_b):
        print('Como la variable pivotal cae en la region critica')
        print('Se rechaza, con una confianza del 95%, la hipotesis nula. Por lo que la varianza NO es 0.020')
    else:
        print('No hay pruebas suficientes para rechazar la hipotesis nula')
        print('Por lo tanto, con una confianza del 95% se puede afirmar que la variaza es 0.020')
    
    print('\n')
    #------ CONTRASTES NO PARAMETRICOS: Prueba de bondad de ajuste ------- #
    #Se realiza el test de hipótesis para verificar si las variables siguen una distribucion normal o no
    print('------PRUEBA DE BONDAD DE AJUSTE-----')
    print('----SE PROBARA SI LAS VARIABLES SIGUEN UNA DISTRIBUCION NORMAL O NO----')
    print('H0: "X sigue una distribucion normal" vs H1: "X NO sigue una distribucion normal".')
    print('\n')

    print('Variable: Paquetes')
    test_normalidad_paquetes = normaltest(df['paquetes'])
    print(test_normalidad_paquetes)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable paquetes sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')

    print('Variable: precio')
    test_normalidad_precio = normaltest(df['precio'])
    print(test_normalidad_precio)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable precio sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')

    print('Variable: ingreso')
    test_normalidad_ingreso = normaltest(df['ingreso'])
    print(test_normalidad_ingreso)
    print('Como el p-valor registrado es mayor que 0.05 podemos afirmar con una confianza del 95% que la variable ingreso sigue una distribucion normal')
    input('Presione una tecla...')
    print('\n')