from difflib import SequenceMatcher
import json
import collections


def find_backbone(json_path):
    with open(json_path) as json_file:
        jdata = json.load(json_file)

    matches = []
    for i_node in range(0, len(jdata['nodes']) - 1):
        node_i1 = jdata['nodes'][i_node]
        node_i2 = jdata['nodes'][i_node+1]
        name1 = (node_i1['name'])
        name2 = (node_i2['name'])

        match = SequenceMatcher(None, name1, name2).find_longest_match(0, name1.find('_'), 0, name2.find('_'))
        matches.append(name1[match.a: match.a + match.size])

    counter = collections.Counter(matches)
    final_match = counter.most_common()[0][0]

    return final_match
