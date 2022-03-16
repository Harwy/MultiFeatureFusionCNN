import pandas as pd
import matplotlib.pyplot as plt

def plot(df1, df2, name="performance"):
    plt.plot(df1['Value'], '-o', label="train", linewidth=2.0)
    plt.plot(df2['Value'], '-o', label="test", linewidth=2.0)
    ax = plt.gca()
    ax.set_ylim([0,10])
    ax.set_aspect(0.6/ax.get_data_ratio())
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Number of epochs", fontsize=20)
    plt.ylabel("Mean absolute error", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(f"./analyse/{name}.jpg", bbox_inches='tight', dpi=400)

if __name__ == "__main__":
    root = "F:/DownLoad/log/dldl/"
    train = root + "run-log-tag-train_MAE.csv"
    test = root + "run-log-tag-valid_MAE.csv"
    name = "dldl_performance"

    df1 = pd.read_csv(train)
    df2 = pd.read_csv(test)
    plot(df1, df2, name)