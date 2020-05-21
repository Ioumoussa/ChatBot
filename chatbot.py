# Idris's chatbot ,In this work I will use Deep NLP to Build my One Chatbot

import numpy as np            
import tensorflow as tns 
import re                #to clean all my text in the training 
import time              # to control the time for training


lignes = open('movie_lines.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding ='utf-8', errors = 'ignore').read().split('\n')

#il faut créer un dico qui sert a mapper les lignes avec leur Id
# il faut découper chaque ligne vu que le premier élément de la ligne correspond à l 'ID le dernier représente la donnée

id_to_line = {}
for line in lignes:
    _line = line.split(' +++$+++ ')
    if(len(_line) == 5):
        id_to_line[_line[0]] = _line[4]

# on va créer une list pour toutes les conversations, on a beaucoupde méta_data, but we should keep juste what is intresting for us, the Ids in the conversation
conversations_id = []
for discussion in conversations[:-1]:
    _conv = discussion.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_id.append(_conv.split(','))     
        
# séparer the inputs = Questions & output = Targets

questions = []
answers = []
for discussion in conversations_id:
    for i in range(len(discussion) - 1):
        questions.append(id_to_line[discussion[i]] )    
        answers.append(id_to_line[discussion[i+1]])
        
        
# Cleaning the Data = make all text in lowercase, remove apostrophes
    
def clean(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it s", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[ - ( ) ' \" # / = @ ; : < ---  > { } + - | . ? , ]", " ", text)
    return text    
         


#cleaning the Questions 
clean_questions = []
for question in questions:
    clean_questions.append(clean(question))

#cleaning the Answers 
clean_ansewrs = []
for answer in answers:
    clean_ansewrs.append(clean(answer))
    
#now we should optimize the trainig, so we need the essential word of  the vocabuary, so we've gonna remove the words
#that's apper less hat 7-9% of the whole corpus   => create a map dico

word_counter = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_counter:
            word_counter[word] = 1
        else:
            word_counter[word] += 1
    

for answer in clean_ansewrs:
    for word in answer.split():
        if word not in word_counter:
            word_counter[word] = 1
        else:
            word_counter[word] += 1            



# now I am in the process of tokenization & Filtering
            
threshold = 20
word_number = 0 # a counter , unique integer that's maps the words, number of occurrences
questions_word_to_integer = {} # dico for mapping the Questions words

for word , count in word_counter.items():
    if count >= threshold:
        questions_word_to_integer[word] = word_number
        word_number += 1
        

answers_word_to_integer = {} # dico for mapping the Answers words    
word_number = 0 # a counter , unique integer that's maps the words, number of occurrences => reinitialisation

for word , count in word_counter.items():
    if count >= threshold:
        answers_word_to_integer[word] = word_number
        word_number += 1



# add the the tokens for the seq2seq model for the last 2 dictionaries, the coder and the decoder
#eos = the end of string | sos = start of string | out = corresond to all the words that were filtred | pad for length
        
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']

for token in tokens:
    questions_word_to_integer[token] = len(questions_word_to_integer) + 1

for token in tokens:
    answers_word_to_integer[token] = len(answers_word_to_integer) + 1
    
        
#create the inverse dictioary  of the answers_word_to_integer dico
# i need the inverse mapping for the implementation of the seq2seq model 
#in Python , i implement a trick to inverse any dictionary , see how :

answers_word_to_integer_to_word = { word_int : w for w , word_int in answers_word_to_integer.items()}   
    

#add the '<EOS>' end of String to the end of every ansewers

for i in range(len(clean_ansewrs)):
    clean_ansewrs[i] += ' <EOS>'


#translating all the questions and the answers into integers
#GOAL = sort al the questions and the answers by their length to optimize the training performance     
#replace all the words that were filtred out by <out>
    
questions_to_int = []
for q in clean_questions:
    integers = []#list of integers each of the integers will be the associated integer to the word in that question  
    for word in q.split():
        if word not in questions_word_to_integer:
            integers.append(questions_word_to_integer['<OUT>'])
        else:
            integers.append(questions_word_to_integer[word])
    questions_to_int.append(integers)
    
                
        

answers_to_int = []
for a in clean_ansewrs:
    integers = []#list of integers each of the integers will be the associated integer to the word in that question  
    for word in a.split():
        if word not in answers_word_to_integer:
            integers.append(answers_word_to_integer['<OUT>'])
        else:
            integers.append(answers_word_to_integer [word])
    answers_to_int.append(integers)
    


# know we should sort the answers and questions lists by the length of the questions
clean_questions_sorted = []
clean_answers_sorted = []

for length in range(1 , 20 + 1):
     for i in enumerate(questions_to_int):
         if len(i[1]) == length:
             clean_questions_sorted.append(questions_to_int[i[0]])
             clean_answers_sorted.append(answers_to_int[i[0]])
             





#***************************************  Part_Two ***** Building The Seq2Seq Model **********************************

# first of all we should cretae the place Holders for our inputs and outputs
             
def model_inputs():
    inputs = tns.placeholder(tns.int32, [None, None], name = 'input')
    targets = tns.placeholder(tns.int32, [None, None], name='target')
    lr = tns.placeholder(tns.float32, name='learning rate')
    keep_prob = tns.placeholder(tns.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob
    
            
#preprocessing the targets
#create batches of 10 answers & add the <SOS> targets at the begining of each answers
#and remove the last element in the answers

def preprocess_targets(targets, //**/, batch_size):
    left_side = tns.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tns.strided_slice(targets, [0,0], [batch_size, -1], [1,                                                                                                                                                            1])
    preprocessed_targets = tns.concat([left_side , right_side], 1) 
    return preprocessed_targets

#create the encoder layer for the RNN Layer
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob,sequence_length):
    lstm =  tns.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tns.contrib.rnn.DropoutWrapper(lstm, input_keep_prob= keep_prob)
    encoder_cell = tns.contrib.rnn.MultiRNNCell([lstm_dropout]* num_layers)
    _, encoder_state = tns.nn.bidirectional_dynamic_rnn(cell_fw= encoder_cell, 
                                                        cell_bw= encoder_cell,
                                                        sequence_length= sequence_length,
                                                        inputs= rnn_inputs,
                                                        dtype = tns.float32)
    return encoder_state


# decoding the training set
# decode of RNN layer
# there are 3 steps to do it
    # step 1 : decode the layer
    # step 2 decode the validation set
    # step 3 : create the decoder
    
def decode_trainig_set(encoder_state, decoder_cell, decoder_embded_input,sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tns.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tns.contrib.seq2seq.prepare_attention(attention_states, attention_option= 'bahdanu', num_units= decoder_cell.output_size)
    training_decoder_function =  tns.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                                attention_keys,
                                                                                attention_values,
                                                                                attention_score_function,
                                                                                attention_construct_function,
                                                                                name = "attn_dec_train")
     
    decoder_output, decoder_final_state,  decode_final_contexte_state = tns.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,  #here it's just the variable decoder_output that interests us, the others c just for visibility.
                                                                                                                               training_decoder_function,
                                                                                                                               decoder_embded_input,
                                                                                                                               sequence_length,
                                                                                                                               scope= decoding_scope)
    decoder_output_dropout = tns.nn.dropout(decoder_ output, keep_prob)
    return output_function(decoder_output_dropout)

#decoding the test validation set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_lenght, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tns.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tns.contrib.seq2seq.prepare_attention(attention_states, attention_option= 'bahdanu', num_units= decoder_cell.output_size)
    test_decoder_function =  tns.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                encoder_state[0], 
                                                                                attention_keys,
                                                                                attention_values,
                                                                                attention_score_function,
                                                                                attention_construct_function,
                                                                                decoder_embeddings_matrix,
                                                                                sos_id, 
                                                                                eos_id, 
                                                                                maximum_lenght, 
                                                                                num_words
                                                                                name = "attn_dec_inf")
     
    test_predictions, decoder_final_state ,  decode_final_contexte_state = tns.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,   #  here it's just the variable test_predictions that interests us, the others c just for visibility.
                                                                                                                   test_decoder_function,
                                                                                                                   scope= decoding_scope)
    return test_predictions

#create the decoder layer for the RNN Layer
def decoder_rnn(decoder_embded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers,word2int, keep_prob, batch_size):
    with tns.variable.scope("decoding_scope") as decoding_scope:
        lstm = tns.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tns.contrib.rnn.DropoutWrapper(lstm, input_keep_prob= keep_prob)
        decoder_cell = tns.contrib.rnn.MultiRNNCell([lstm_dropout]* num_layers) 
           
        weights = tns.truncated_normal_initializer(stddev = 0.1)
        biases = tns.zeros_initializer()
        output_function = lambda x : tns.contrib.layers.fully_connected(x,
                                                                        num_words,
                                                                        None,
                                                                        scope= decoding_scope,
                                                                        weights_initializers = weights,
                                                                        biases_initializers = biases )

        training_predictions = decode_trainig_set(encoder_state,
                                                  decoder_cell, 
                                                  decoder_embded_input, 
                                                  sequence_length,
                                                  decoding_scope, 
                                                  output_function, 
                                                  keep_prob, 
                                                  batch_size) 

        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                             decoder_cell,
                                             decoder_embeddings_matrix,
                                             word2int['<SOS>'],
                                             word2int['<EOS>'],
                                             sequence_length - 1,
                                             num_words,
                                             decoding_scope,
                                             output_function,
                                            keep_prob,
                                            batch_size)
    return training_predictions, test_predictions   


# building the Seq2Seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, questions_num_words, rnn_size, num_layers, questions_word_to_integer):
    encoder_embedded_input = tns.contrib.layers.embed_sequence(inputs, 
                                                               answers_num_words + 1,
                                                               encoder_embedding_size,
                                                               initializer = random_uniform_initializer(0, 1)
                                                               )
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocessed_targets(targets, questions_word_to_integer, batch_size)
    decoder_embeddings_matrix = tns.variable(tns.random_uniform([questions_num_words + 1 , decoder_embedding_size],0,1))
    decoder_embedded_input = tns.nn.embedding_lookup( decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_word_to_integer,
                                                         keep_prob,
                                                         batch_size)

    return training_predictions, test_predictions



########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########

# hyperparametrs
 
epochs = 50
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5 

# defining a session , create a tensorflow variable, but we have to reset the graph first
tns.reset_default_graph()
session = tns.InteractiveSession()

#loading the model inputs
inputs, targets , lr, keep_prob = model_inputs()

#setting the sequence lenght

sequence_length = tns.placeholder_with_default(25, None, name ='Sequence Length')

#getting the shape of the inputs tensor
input_shape = tns.shape(inputs)

#getting the trainig and the testing predictions
#we must get the training&test predections when we feeding the model with the inputs loaded in the line 352
training_predictions, test_predictions = seq2seq_model(tns.reverse(inputs, [-1]),
                                                                   targets,
                                                                   keep_prob,
                                                                   batch_size,
                                                                   sequence_length,
                                                                   len(answers_word_to_integer),
                                                                   len(questions_word_to_integer),
                                                                   encoding_embedding_size,
                                                                   decoder_embedding_size,
                                                                   decoding_embedding_size,
                                                                   rnn_size,
                                                                   num_layers,
                                                                   questions_word_to_integer)

#setting up the loss error the optimizer, and the gradient Clipping
with tns.name_scope("optimization"):
    loss_error = tns.contrib.seq2seq.sequence_loss(training_predictions,
                                                   targets,
                                                   tns.ones([input_shape[0], sequence_length]))

    optimizer = tns.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence)  for sequences in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

#spliting the Data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_word_to_integer))
        padded_answers_in_batch = np.array(apply_padding(answerns_in_batch, answers_word_to_integer))
        yield padded_questions_in_batch, padded_answers_in_batch

#splitting into training & validation sets
training_validation_split = int(len(clean_questions_sorted) * 0.15)
training_questions =   clean_questions_sorted[training_validation_split:]
training_answers =  clean_answers_sorted[training_validation_split:]

validation_questions =   clean_questions_sorted[:training_validation_split]
validation_answers =  clean_answers_sorted[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ( (len(training_questions)) // batch_size  // 2)
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())

for epoch in range(1, epochs +1 ):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = ()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
 
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error

            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop
                    break
            
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")

########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########

checkpoint = "./chatbot_weights.ckpt" # first we should load the weights from the file that store it
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

def convert_string2int(question, word2int):
    question = clean(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


while(True):
    question = input(" Me : ")
    if question == 'Bye'
        break
    question = convert_string2int(questions_word_to_integer)
    question = question + [questions_word_to_integer['<PAD>']*  (20 - len(question))]
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answers_word_to_integer_to_word[i] == 'i':       
            token = 'I'
        elif answers_word_to_integer_to_word[i] == '<EOS>':
            token = '.'
        elif answers_word_to_integer_to_word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answers_word_to_integer_to_word[i]
        answer += token

        if token == '.'
            break
  
    print('MyChatbot '+ answer) 














