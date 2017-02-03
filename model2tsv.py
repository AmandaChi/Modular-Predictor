import sys
from keras.models import load_model
folder = ".//adintent_model_tf/"
model = load_model(sys.argv[1])
model.summary()

def printLayer(modelLayer):
    for layer in modelLayer.layers:
        if(layer.name[0:5] == "merge" or layer.name[0:10] == "sequential"):
            printLayer(layer)
        else:
            print(layer.name)
            weights = layer.get_weights()
            #Dense Layer
            if layer.name[0:5] == "dense":
                of = open(folder + layer.name + ".txt",'w')
                of.write(str(weights[0].shape[0]) + " " + str(weights[0].shape[1])+"\n")
            # W
                for i in range(0,weights[0].shape[0]):
                    for j in range(0,weights[0].shape[1]):
                        of.write(str(weights[0][i][j]) + " ")
                    of.write("\n")
                of.write("\n")
            # b
                for i in range(0,weights[1].shape[0]):
                    of.write(str(weights[1][i]) + " ")
                of.write("\n")
            #Embedding Layer
            if layer.name[0:9] == "embedding":
            #print(len(weights))
            #print(weights[0].shape)
                of = open(folder + layer.name + ".txt",'w')
                # Embs
                of.write(str(weights[0].shape[0]) + " " + str(weights[0].shape[1])+"\n")
                for i in range(0,weights[0].shape[0]):
                    for j in range(0,weights[0].shape[1]):
                        of.write(str(weights[0][i][j]) + " ")
                    of.write("\n")
            #Conv1D Layer
            if layer.name[0:13] == "convolution1d":
                #print(len(weights))
                #print(weights[0].shape)
                #print(weights[1].shape)
                of = open(folder + layer.name + ".txt",'w')
                of.write(str(weights[0].shape[0]) + " " + str(weights[0].shape[2]) + " " + str(weights[0].shape[3]) +"\n")
                #W
                for i in range(0,weights[0].shape[0]):
                    for j in range(0,weights[0].shape[2]):
                        for k in range(0,weights[0].shape[3]):
                            of.write(str(weights[0][i][0][j][k]) + " ")
                        of.write("\n")
                    of.write("\n")
                of.write("\n")
                #b
                #print(weights[1].shape)
                for i in range(0,weights[1].shape[0]):
                    of.write(str(weights[1][i]) + " ")
                of.write("\n")
            of.close()

printLayer(model)     
#    weights = layer.get_weights()
    #Dense Layer
#    if layer.name[0:5] == "dense":
#        of = open(layer.name + ".txt",'w')
#        # W
#        for i in range(0,weights[0].shape[0]):
#            for j in range(0,weights[0].shape[1]):
#                of.write(str(weights[0][i][j]) + " ")
#            of.write("\n")
#        of.write("\n")
#        # b
#        print(weights[1].shape)
#        for i in range(0,weights[1].shape[0]):
#            of.write(str(weights[1][i]) + " ")
#        of.write("\n")
    #Embedding Layer
#    if layer.name[0:9] == "embedding":
#        print(len(weights))
#        print(weights[0].shape)
#        of = open(layer.name + ".txt",'w')
#        # Embs
#        for i in range(0,weights[0].shape[0]):
#            for j in range(0,weights[0].shape[1]):
#                of.write(str(weights[0][i][j]) + " ")
#            of.write("\n")


#of.close()
