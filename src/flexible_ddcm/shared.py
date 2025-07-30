def pandas_dot(df, series):
    return df[list(series.index.values)].dot(series)


def build_covariates(df, covariates):
    df = df.copy()
    for col, definition in covariates.items():
        if col in df:
            continue
        df[col] = df.eval(definition)
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    return df


def build_list_of_grids(model_options):
    states = {}
    # Should just provide funtions. That are labelled ín
    for state, options in model_options["state_space"].items():
        if options["type"] == "list":
            states[state] = options["list"]
        elif options["type"] == "integer_grid":
            states[state] = list(range(options["lowest"], options["highest"]))
    return states