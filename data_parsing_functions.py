import csv


def get_sequence_lengths_and_percentages():
    with open("uniprot_data1.csv", 'r') as file:
        csvreader = csv.reader(file)
        sequences = []
        sequence_lengths = []
        percentages = []
        for row in csvreader:
            sequences.append(row[4])
            percentages.append(row[5:])
        for sequence in sequences:
            sequence_lengths.append(len(sequence.strip()))
    return sequence_lengths, percentages


def get_weighted_matrix(sequence_lengths, percentages):
    weight_matrix = percentages
    max_sequence_length = sequence_lengths[0]
    for length in sequence_lengths:
        if length > max_sequence_length:
            max_sequence_length = length
    for i in range(len(percentages)):
        for j in range(len(percentages[i])):
            weight_matrix[i][j] = round(((100 - (float)(weight_matrix[i][j])) / 100) * sequence_lengths[i]
                                        * max_sequence_length / 100, 2)
    return weight_matrix

    # How to call the functions:
    # sequence_lengths, percentages = get_sequence_lengths_and_percentages()
    # weight_matrix = get_weighted_matrix(sequence_lengths, percentages))
