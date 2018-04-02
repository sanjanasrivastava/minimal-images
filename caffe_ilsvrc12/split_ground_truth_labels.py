import json


def split_labels(dataset):

    labels_filename = dataset + '.txt'

    with open(labels_filename) as labels_file:
        labels = labels_file.readlines()
        labels = map(lambda x: x.split(), labels)
        labels = {label[0][:-5]: int(label[1]) for label in labels}

    with open(dataset + '-labels.json', 'w') as fp:
        json.dump(labels, fp)


split_labels('val')


