from pathlib import Path
import random
import collections
from math import log
import matplotlib.pyplot as plt

#Partie 1 : Dataset

def generate_examples(name_directory):
    #création d'un objet path correspondant au chemin vers mon corpus
    chemin = Path(name_directory)
    #liste de tuples (liste_mots, etiquette) indiquant pour chaque mail si c'est un spam ou ham
    liste_mails = list()
    nb_spam = 0
    nb_ham = 0
    #boucle qui parcourt tous les chemins (récursivement) se finissant par .txt
    for path in chemin.glob('**/*.txt'):
        fichier = open(str(path),"r")
        contenu = fichier.read()
        liste_mots = contenu.split(" ")
        if "spm" in path.name:
            liste_mails.append((liste_mots, "spam"))
            nb_spam = nb_spam + 1
        else:
            liste_mails.append((liste_mots, "ham"))
            nb_ham = nb_ham + 1
    print("The corpus has " + str(len(liste_mails)) + " examples.")
    print("It contains " + str(nb_ham) + " hams et " + str(nb_spam) + " spams.")
    return liste_mails

liste_mails = generate_examples("D:\lingspam_public\lemm_stop")
#liste_mails = generate_examples('/media/utilisateur/RED/lingspam_public/bare')


#Separer aleatoirement la liste en trainning set et test set, la fonction retourne un tuple avec les deux sets
#Pourcentage_train doit être un float compris entre 0 et 1
def split_corpus(liste_mails, pourcentage_train):
    random.shuffle(liste_mails)
    train_set = liste_mails[:int(len(liste_mails)*pourcentage_train)]
    test_set = liste_mails[int(len(liste_mails)*pourcentage_train):]
    return (train_set, test_set)

(train_set, test_set) = split_corpus(liste_mails, 0.8)




#Partie 2 : Naive Bayes classifier

#    log(p(y|x)) ∝ log( p(y) · produit_pour_i_allant_de_1_a_k(p(xi|y)) )
#<=> log(p(y|x)) ∝ log(p(y)) + log(produit_pour_i_allant_de_1_a_k(p(xi|y)))
#<=> log(p(y|x)) ∝ log(p(y)) + somme_pour_i_allant_de_1_a_k(log(p(xi|y)))
#We can use log p(y|x) in the MAP decision rule instead of p(y|x) because log p(y|x) is proportional to p(y|x).

# takes as input a list of examples and returns a dictionary of dictionaries
# that maps every pairs (word, label) to the number of occurrences of word in mails labeled by label
def map(list_examples):
    dictionnaire = collections.defaultdict(lambda:collections.defaultdict(float))
    for (liste_mots, etiquette) in list_examples:
        for mot in liste_mots:
            dictionnaire[mot][etiquette] = dictionnaire[mot][etiquette] + 1
    return dictionnaire


class Classifier:
    # cette fonction estime les parametres du naive bayes classifier (correspond à fit)
    # elle calcule p(ham), p(spam), p(mot|etiquette) pour tout mot et etiquette
    def __init__(self, train_set):
        self.train_set = train_set
        nb_mails = len(train_set)
        # calcul nombre de hams et spams et nombre de mots dans les spams et les hams
        nb_hams = 0
        nb_spams = 0
        mots_dans_hams = 0
        mots_dans_spams = 0
        for (mail, etiquette) in train_set:
            if etiquette == "ham":
                nb_hams = nb_hams+1
                mots_dans_hams = mots_dans_hams + len(mail)
            if etiquette == "spam":
                nb_spams = nb_spams+1
                mots_dans_spams = mots_dans_spams + len(mail)
        self.log_p_ham = log(nb_hams/nb_mails)
        self.log_p_spam = log(nb_spams/nb_mails)
        # log_prob de chaque mot sachant etiquette (permet de calculer log_prob[mot][etiquette])
        self.log_prob = collections.defaultdict(lambda:collections.defaultdict(float))
        occurences = map(train_set)
        for entree in occurences.items():
            # entree[0] est le mot et entree[1]['ham'] est le nombre d'occurences du mot dans des hams
            self.log_prob[entree[0]]["ham"] = log((entree[1]["ham"]+1)/mots_dans_hams)
            self.log_prob[entree[0]]["spam"] = log((entree[1]["spam"]+1)/mots_dans_spams)

    # cette fonction predit pour une liste de mots s'il s'agit d'un spam ou d'un ham
    def predict(self, liste_mots):
        # log(p(ham|x)) ∝ log(p(ham)) + somme_pour_i_allant_de_1_a_k(log(p(xi|ham)))
        # log(p(spam|x)) ∝ log(p(spam)) + somme_pour_i_allant_de_1_a_k(log(p(xi|spam)))
        somme_p_mots_ham = 0
        somme_p_mots_spam = 0
        for mot in liste_mots:
            somme_p_mots_ham = somme_p_mots_ham + self.log_prob[mot]["ham"]
            somme_p_mots_spam = somme_p_mots_spam + self.log_prob[mot]["spam"]
        log_prob_ham = self.log_p_ham + somme_p_mots_ham
        log_prob_spam = self.log_p_spam + somme_p_mots_spam
        if log_prob_ham > log_prob_spam:
            return "ham"
        else:
            return "spam"
    # cette fonction prend en parametre une liste d'exemples annotes et retourne la proportion de mails correctement predits (accuracy)
    def score(self, liste_exemples):
        # liste_exemple est une liste de tuples (mail, etiquette)
        nb_exemples = len(liste_exemples)
        res_corrects = 0
        for (mail, etiquette) in liste_exemples:
            if (self.predict(mail) == etiquette):
                res_corrects = res_corrects + 1
        accuracy = res_corrects/nb_exemples
        # le resultat est un nombre compris entre 0 et 1
        return accuracy

c = Classifier(train_set)
print("\nThe classifier has an accuracy of " + str(c.score(test_set)*100) + "%.")




#Partie 3 : Evaluation

# la matrice de confusion est un tableau de 2x2 tel que la case i,j 
# contient le nombre de mails ayant pour etiquette i qui ont été classifés comme des j
def conf_matrix_gen(liste_exemples, classifieur):
    confusion_matrix = [[0,0],[0,0]]
    for (mail, etiquette) in liste_exemples:
        if (etiquette == "ham"):
            i = 0
        else:
            i = 1
        if (classifieur.predict(mail) == "ham"):
            j = 0
        else:
            j = 1
        confusion_matrix[i][j] += 1
    return confusion_matrix

mat = conf_matrix_gen(test_set, c)
print("\nConfusion matrix:\n\t ham \t spam \nham\t" + str(mat[0][0]) + "\t\t" + str(mat[0][1]) + "\nspam\t" + str(mat[1][0]) + "\t" + str(mat[1][1]))

# the confusion gives us a better idea of where the classifier is making mistakes,
# this is useful to know because some mistakes are worse than others 
# (classifying ham as spam is worse than letting a spam pass through)

# plotting the classifier performance with respect to the size of the train set
plt.title("Evolution of the classifier's performance according to the size of the train set")
plt.xlabel("Percentage of the corpus used for training")
plt.ylabel("Accuracy of the classifier")

xpoints = list()
ypoints = list()
for i in range(10, 100, 10):
    (train_set, test_set) = split_corpus(liste_mails, i/100)
    c = Classifier(train_set)
    xpoints.append(i)
    ypoints.append(c.score(test_set)*100)

plt.plot(xpoints, ypoints)
plt.show()

# The accuracy of the classifier tends to increase along with the size of the training set.
# This is to be expected: the more data it has to learn from, the more precise the parameters it calcultes
# However, when the train set is too big (90%), we sometimes get strange results because the test set is too small to be representative.

# plotting the accuracies of  100 train-test splits (all with a 80:20 proportion)
plt.title("Accuracies of 100 random train-test splits (all with proportion 80:20)")
plt.ylabel("Accuracy")

xpoints = list()
ypoints = list()
min_accuracy = 100
max_accuracy = 0
for i in range(100):
    (train_set, test_set) = split_corpus(liste_mails, 0.8)
    c = Classifier(train_set)
    accuracy = c.score(test_set)*100
    if accuracy < min_accuracy:
        min_accuracy = accuracy
    if accuracy > max_accuracy:
        max_accuracy = accuracy
    xpoints.append(i)
    ypoints.append(accuracy)
    
print("\nThe highest accuracy found was " + str(max_accuracy))
print("The lowest accuracy found was " + str(min_accuracy))
    
plt.scatter(xpoints, ypoints)
plt.show()

# The accuracy of all of the classifiers is quite high (above 90%). The program seems rather satisfying.

print("\nTen words with the highest probability of being in ham:")
c = Classifier(liste_mails)
for i in range(10):
    max_proba_ham = -1000
    max_mot = "                                "
    for entree in c.log_prob.items():
      if entree[1]["ham"] > max_proba_ham:
          max_proba_ham = entree[1]["ham"]
          max_mot = entree[0]
    c.log_prob.pop(max_mot)
    print(str(i+1) + ": " + max_mot)

print("\nTen words with the highest probability of being in spam:")
c = Classifier(liste_mails)
for i in range(10):
    max_proba_spam = -1000
    max_mot = "                                       "
    for entree in c.log_prob.items():
      if entree[1]["spam"] > max_proba_spam:
          max_proba_spam = entree[1]["spam"] 
          max_mot = entree[0]
    c.log_prob.pop(max_mot)
    print(str(i+1) + ": " + max_mot)

# Les mots avec la probabilité la plus élevée d'être dans des spams sont aussi ceux qui ont le plus de chance
# d'être dans des hams. Il s'agit plutôt de signes de ponctuation que de mots. Peut-être que ce genre de signes
# ne devraient pas être pris en compte, puisqu'ils sont présents dans les deux cas. Le classifieur pourrait donc
# encore être ammélioré