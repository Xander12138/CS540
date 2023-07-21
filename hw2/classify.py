import os
import math
# discussed idea with Guanlin Chen

#These first two functions require os operations and so are completed for you
#Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

#Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f,'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


#The rest of the functions need modifications ------------------------------
#Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # with open(filepath) as f:
    #     lines = f.read()
    #     for line in lines.split('\n'):
    #         if line in vocab:
    #             if line in bow:
    #                 bow[line] += 1
    #             else:
    #                 bow[line] = 1
    #         else:
    #             if None in bow:
    #                 bow[None] += 1
    #             else:
    #                 bow[None] = 1
    # f.close()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line in vocab:
                if line in bow:
                    bow[line] += 1
                else:
                    bow[line] = 1
            else:
                if None in bow:
                    bow[None] += 1
                else:
                    bow[None] = 1
    return bow


#Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1 # smoothing factor
    logprob = {}
    logprob[label_list[0]] = 0
    logprob[label_list[1]] = 0
    y2020 = 0
    y2016 = 0
    for list in training_data:
        if list['label'] == '2020':
            y2020 += 1
        else:
            y2016 += 1
    lab2020 = (y2020 + smooth) / (y2016 + y2020 + 2)
    lab2016 = (y2016 + smooth) / (y2016 + y2020 + 2)
    logprob['2020'] = math.log(lab2020)
    logprob['2016'] = math.log(lab2016)
    return logprob

#Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    smooth = 1 # smoothing factor
    word_prob = {}
    wc = 0
    tempData = {}

    tempData[None] = 0
    for word in vocab:
        tempData[word] = 0

    for data in training_data:
        if data['label'] == label:
            for word in data['bow']:
                wc += data['bow'][word]

    for data in training_data:
        if data['label'] == label:
            for word in data['bow']:
                if word in vocab:
                    tempData[word] += data['bow'][word]
                else:
                    tempData[None] += data['bow'][word]
        else:
            continue

    for word in tempData:
        word_prob[word] = math.log((tempData[word] + smooth) / (wc + smooth * (len(vocab) + 1)))

    return word_prob

##################################################################################
#Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    lab1 = p_word_given_label(vocab, training_data, label_list[0])
    lab2 = p_word_given_label(vocab, training_data, label_list[1])
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(training_data, label_list)
    retval['log p(w|y=2020)'] = lab1
    retval['log p(w|y=2016)'] = lab2
    return retval


#Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """

    retval = {}
    retval['predicted y'] = ' '
    p2020 = 0
    p2016 = 0

    bow = create_bow(model['vocabulary'], filepath)
    for line in bow:
        if line in model['log p(w|y=2016)']:
            p2016 += (model['log p(w|y=2016)'][line] * bow[line])
        else:
            p2016 += (model['log p(w|y=2016)'][None])

        if line in model['log p(w|y=2020)']:
            p2020 += (model['log p(w|y=2020)'][line] * bow[line])
        else:
            p2020 += (model['log p(w|y=2016)'][None])


    retval['log p(y=2016|x)'] = p2016 + model['log prior']['2016']
    retval['log p(y=2020|x)'] = p2020 + model['log prior']['2020']

    if retval['log p(y=2020|x)'] > retval['log p(y=2016|x)']:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    return retval
