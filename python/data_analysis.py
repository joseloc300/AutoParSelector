import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt


def analyze_data(loop_features, loop_targets, data_analysis_params, pre_data_processing):
    if (pre_data_processing and not (data_analysis_params["pre_data_processing"])) or \
            (not pre_data_processing and not (data_analysis_params["post_data_processing"])):
        return

    print("+ Data Analysis")
    if data_analysis_params["data_correlation"]["enabled"]:
        data_correlation(loop_features, loop_targets, data_analysis_params["data_correlation"])


# testar scoring com todos, só NAS e só Polybench - DONE
# grandes diferenças estruturais entre NAS e polybench, especializar em 2?
# ter em conta o impacto da extraçao de features (para uso futuro) se estáticas têm desempeho semelhante a dinamicas
# pode n fazer sentido usar as dinamicas (por ex.)
# fazer data analysis para grupo restrito de features (as 10 que esperamos ter melhores resultados por ex) - DONE
# ver correlaçao features/target (desejavel != 0) - DONE
# analisar valores de maneira nao visual (pesquisa certo ranges de valores) - DONE
# tentar criar clusteres de vars ou selecionar vars principais
def data_correlation(loop_features, loop_targets, correlation_params):
    print("++ Data correlation")

    corr_matrix = get_correlation_matrix(loop_features, loop_targets)
    if correlation_params["graphical_results"]:
        sn.heatmap(corr_matrix, annot=True)
        plt.show()
    else:
        print(corr_matrix)


def get_correlation_matrix(loop_features, loop_targets):
    converted_data = convert_data_frame_format(loop_features, loop_targets)
    df = pd.DataFrame(data=converted_data)
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.fillna(0)

    return corr_matrix


def convert_data_frame_format(loop_features, loop_targets):
    numpy_array = np.array(loop_features)
    transpose = numpy_array.T
    transpose = np.append(transpose, [loop_targets], axis=0)
    transpose_array = transpose.tolist()

    converted_data = {}
    columns = []

    for i in range(len(transpose_array)):
        converted_data[str(i)] = transpose_array[i]
        columns.append(str(i))

    return converted_data
