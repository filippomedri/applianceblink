from FridgeStateClassifier import FridgeStateClassifier as FSC


def main():
    fsc = FSC()

    print('extract')
    fsc.extract('data_59_all.csv')

    print('build model')
    fsc.build_model_4()

    print('run')
    fsc.run_classifiers()

    print('get scores')
    print(fsc.evaluate_classifiers())

    print('plot roc curves')
    fsc.plot_roc_curve_classifiers()

    print('get AUC')
    fsc.get_auc()

if __name__ == "__main__":
    main()

