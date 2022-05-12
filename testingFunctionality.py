def VisualizeData():

    cat_features = []
    num_features = []

    fig, ax = plt.subplots(6, 6)
    coordx = 0
    coordy = 0

    # DATA VISUALIZATION
    for column in df.columns:
        # print("coordx", coordx, "coordy", coordy)
        if coordx == 4 and coordy == 6:
            break
        elif coordx == 6:
            coordx = 0
            coordy += 1

        frequency = df[column].value_counts(dropna=False)

        # plt.subplot(1, 2, 1)
        if df[column].nunique() <= 7:
            cat_features.append(df[column])
            sns.barplot(
                x=frequency.index, y=frequency.values, alpha=0.8, ax=ax[coordx, coordy]
            )
            coordx += 1

        else:
            num_features.append(df[column])
            maxF = max(frequency.values)
            minF = min(frequency.values)
            barWidth = (maxF - minF) / 15
            sns.set_style("whitegrid")
            sns.distplot(
                df[column].dropna(),
                kde=False,
                color="red",
                bins=15,
                ax=ax[coordx, coordy],
            )
            coordx += 1
        # sns.barplot
        # sns.pointplot

        plt.title("Distribution of values for column " + column)
        plt.ylabel("Number of Occurrences", fontsize=12)
        plt.xlabel("Values", fontsize=12)

        """ @deprecated - DIAGRAMMA A TORTA PER VALORI NULLI
        nnv = 0
        nv = 0
        for i in range(0, len(frequency.values)):
            if not pd.isna(frequency.index.values[i]):
                nnv += frequency.values[i]
            else:
                nv += frequency.values[i]

        pielist = [(nnv / (nnv + nv)), (nv / (nnv + nv))]
        colors = plt.get_cmap("Blues")(np.linspace(0.4, 0.9, len(pielist)))
        print(pielist)

        
        print(
            "frequency.index of lenght",
            frequency.index.values.size,
            " : ",
            frequency.index.values,
            " frequency.values: of lenght",
            frequency.values.size,
            " : ",
            frequency.values,
        )
        plt.subplot(1, 2, 2)
        plt.pie(
            pielist,  # frequency.values,
            colors=colors,
            radius=3,
            center=(4, 4),
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            frame=True,
        )"""

    plt.show()


# ------------------------------------------------------------------------------------------------------


def visualizeCategorical():
    counter = 0

    for cat in cat_features:
        coordx = 0
        coordy = 0

        frequency = df[cat].value_counts(dropna=False)

        if counter < 9:
            fig, ax = plt.subplots(3, 3)
        elif counter == 9:
            counter = 0

        for i in range(0, 3):
            for j in range(0, 3):
                sns.barplot(
                    x=frequency.index,
                    y=frequency.values,
                    alpha=0.8,
                    ax=ax[coordx, coordy],
                )
                coordx += 1
            coordy += 1

        counter += 1
