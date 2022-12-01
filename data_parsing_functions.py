import csv


def get_sequence_lengths_and_percentages(csvreader):
    sequences = []
    sequence_lengths = []
    percentages = []
    species = []
    for row in csvreader:
        species.append(row[2])
        sequences.append(row[4])
        percentages.append(row[5:])
    for sequence in sequences:
        tempLen = 0
        for char in sequence:
            if char != "-":
                tempLen += 1
        sequence_lengths.append(tempLen)
    return sequence_lengths, percentages, species


def get_weighted_matrix(sequence_lengths, percentages):
    weight_matrix = percentages
    max_sequence_length = sequence_lengths[0]
    for length in sequence_lengths:
        if length > max_sequence_length:
            max_sequence_length = length
    for i in range(len(percentages)):
        for j in range(len(percentages[i])):
            weight_matrix[i][j] = (100 - (float)(weight_matrix[i][j])) * max_sequence_length // 100
    return weight_matrix

    # How to call the functions:
    # sequence_lengths, percentages = get_sequence_lengths_and_percentages()
    # weight_matrix = get_weighted_matrix(sequence_lengths, percentages))
