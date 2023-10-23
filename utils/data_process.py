from js import document, console, Uint8Array, window, File


def get_feature_types(columns: []):
    numerical = []
    ordinal = []
    nominal = []
    for item in columns:
        try:
            value = int(document.getElementById(str(item)).value)
            if value == 1:
                numerical.append(item)
            elif value == 2:
                nominal.append(item)
            elif value == 3:
                ordinal.append(item)
        except:
            print('An exception occurred', item)

    return numerical, ordinal, nominal
