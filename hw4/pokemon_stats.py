import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    pokemons = []
    with open(filepath) as csvfile:
        reader = csv.DictReader(csvfile)
        i = 0
        for row in reader:
            if i < 20:
                temp = row
                del temp['Legendary']
                del temp['Generation']
                pokemons.append(temp)
                i += 1
                continue
            break
    csvfile.close()
    for pokemon in pokemons:
        for stat in pokemon:
            if not(stat == 'Name' or stat == 'Type 1' or stat == 'Type 2'):
                pokemon[stat] = int(pokemon[stat])
    return pokemons


def calculate_x_y(stats):
    offensive = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    defensive = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    xy = (offensive, defensive)
    return xy


def hac(dataset):
    clusters = []

    for i in range(len(dataset)):
        if dataset[i]:
            clusters.append({'id': i, 'data': [dataset[i]]})
        else:
            del dataset[i]

    cluster_id = len(clusters)
    Z = []
    shortest_idx = [-1, -1]

    while len(clusters) > 1:
        shortest_distance = 100000000000

        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                curr_distance = cluster_distance(clusters[i]['data'], clusters[j]['data'])
                if curr_distance < shortest_distance:
                    shortest_distance = curr_distance
                    shortest_idx = [i, j]
                else:
                    continue

        i = shortest_idx[0]
        j = shortest_idx[1]
        clusters[i]['data'].extend(clusters[j]['data'])
        level = len(clusters[i]['data'])
        Z.append([min(clusters[i]['id'], clusters[j]['id']),
                  max(clusters[i]['id'], clusters[j]['id']), shortest_distance, level])
        del clusters[j]
        clusters[i]['id'] = cluster_id
        cluster_id += 1

    return np.array(Z)


def random_x_y(m):
    random_pokemon = []
    for i in range(m):
        random_pokemon.append((random.randint(1, 359), random.randint(1, 359)))
    return random_pokemon


def imshow_hac(dataset):
    clusters = []
    for i in range(len(dataset)):
        if dataset[i]:
            plt.scatter(dataset[i][0], dataset[i][1])
            clusters.append({'id': i, 'data': [dataset[i]]})
        else:
            del dataset[i]
    plt.pause(0.1)
    cluster_id = len(clusters)
    Z = []
    shortest_idx = [-1, -1]
    short = []

    while len(clusters) > 1:
        shortest_distance = 100000000000
        for i in range(len(clusters) - 1):
            for j in range(i + 1, len(clusters)):
                short_distance = 1000000000000
                for particle_1 in clusters[i]['data']:
                    for particle_2 in clusters[j]['data']:
                        curr_distance = math.sqrt(
                            (particle_2[0] - particle_1[0]) ** 2 + (particle_2[1] - particle_1[1]) ** 2)
                        if curr_distance < short_distance:
                            short_distance = curr_distance
                            part1 = particle_1
                            part2 = particle_2
                curr = short_distance
                if curr < shortest_distance:
                    shortest_distance = curr
                    short_position = [part1, part2]
                    shortest_idx = [i, j]
                else:
                    continue

            if short_position not in short:
                short.append(short_position)

        i = shortest_idx[0]
        j = shortest_idx[1]
        clusters[i]['data'].extend(clusters[j]['data'])
        level = len(clusters[i]['data'])
        Z.append(
            [min(clusters[i]['id'], clusters[j]['id']), max(clusters[i]['id'], clusters[j]['id']), shortest_distance,
             level])
        del clusters[j]
        clusters[i]['id'] = cluster_id
        cluster_id += 1

    for point in short:
        plt.plot([point[0][0], point[1][0]], [point[0][1], point[1][1]])
        plt.pause(0.1)
    plt.show()

def cluster_distance(cluster_1, cluster_2):
    short_distance = 1000000000000
    for partical_1 in cluster_1:
        for partical_2 in cluster_2:
            curr_distance = math.sqrt((partical_2[0] - partical_1[0]) ** 2 + (partical_2[1] - partical_1[1]) ** 2)
            if curr_distance < short_distance:
                short_distance = curr_distance
    return short_distance


if __name__ == "__main__":
    pokemons = load_data("Pokemon.csv")
    # print(len(pokemons))
    # print(pokemons)
    pokemons_x_y = []
    for row in pokemons:
        pokemons_x_y.append(calculate_x_y(row))

    # print(hac(pokemons_x_y))
    randomxy = random_x_y(20)
    imshow_hac(pokemons_x_y)
