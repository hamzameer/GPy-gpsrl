import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import poisson
from scipy.stats import fisk

import random
from itertools import combinations

from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest

import GPy



## Core GPSRL functions

# partition data on basis of rule list 
def partitiondata(rulelist, data, test = 0, minth = 60):
    lr = len(rulelist)
    rl = []
    partition = {}
    if test == 1: s = 0
    else: s = 2
    not_seen = (data.iloc[:,0] == 'abc') == False # oldest = EVERYTHING false
    idx = 0
    for rule_idx, rule in enumerate(rulelist): 
        conditional = data.eval(rule) # all elements that follow rule
        newcondition = conditional & not_seen
        if sum(newcondition) > minth:
            partition[idx] = {}
            partition[idx]['conditional'] = newcondition # follow rule AND not in old
            not_seen = ~(conditional | ~not_seen)
            partition[idx]['rule'] = rule
            partition[idx]['support'] = sum(newcondition)
            rl.append(rule)
            idx+=1
    lp = len(partition)
    partition[lp] = {}
    partition[lp]['conditional'] = not_seen
    partition[lp]['rule'] = 'Otherwise'
    partition[lp]['support'] = sum(not_seen)
    partition['rulelist'] = rl
    partition['numparts'] = lp+1
    partition['rulesnotused'] = [r for r in rulelist if r not in rl ]
    return partition


# propose a new rule list
def proposenewlist(rulelist, Acopy, lenlist):

    # Acopy: the list of all rules excluding dt
    # lenlist: length of the rulelist
    # TODO
    # Return: new rulelist and qratio and indx of change
    
    # If lenlist = 0 - then add 
    # If lenlist = 1 - then add or remove
    # otherwise - either add, remove or move
    
    newlist = rulelist.copy()
    if lenlist  == 0:
        proposal = 0
    elif lenlist == 1:
        proposal =  random.choice([0,1])
    else:
        proposal = random.choice([0,1,2])
        
    if proposal == 0: # add
        # rule that is added
        newrule = Acopy.pop(random.randrange(len(Acopy)))
        # index where it is added
        indx1 = random.randint(0,lenlist)
        newlist.insert(indx1,newrule)
#         newlist = oldlist + [newrule]
#         random.shuffle(newlist)
        qratio = 1/(len(Acopy) - len(rulelist))
        ind = indx1
        
    elif proposal == 1:  # subtract
        # index where it is subtracted
        indx1 = random.randint(0,lenlist-1)
        newlist.pop(indx1)
#         newlist.pop(random.randrange(len(oldlist)))
#         newlist = oldlist 
        qratio = 1/(len(Acopy) - len(rulelist)-1)
        ind = indx1
        
    elif proposal == 2:  # move
        # which is moving 
        indx1 = random.randint(0,lenlist-1)
        # to where is it moving
        indx2 = random.randint(0,lenlist-1)
        newlist.insert(indx2, newlist.pop(indx1))
#         random.shuffle(oldlist)
#         newlist = oldlist 
        ind = min(indx1,indx2)
        qratio = 1 
        
    return newlist, qratio, ind

# probability depends on number as well as cardinality
def probrulelist2(rulelist, meanrules, card_dict, numrules):
    cardcopy = card_dict.copy()
    nlist = len(rulelist)
    cards = [len(rule.split('&')) for rule in rulelist]
    p = poisson.pmf(nlist, meanrules)
    for i in range(len(rulelist)):
        card = cards[i]
        if card == 1: pcard = 1/6 
        elif card == 2: pcard = 2/3
        else: pcard = 1/6
        prule = 1/cardcopy[card]
        cardcopy[card] -= 1 
        p = p*pcard*prule
    return p

# function to run the mcmc chain
def mcmcchain(niter, rulelistold, partitionold, A, data, modold, loglikold, Dtest, 
              iterstart = 0,
              dlist = [], card_dict = [], meanrules = 3, acceptancelist = []):
    
    numrules = len(A)
    Acopy = A.copy()
    #niter = 2000
    lenlist = len(rulelistold)
    probold = probrulelist2(rulelistold, meanrules, card_dict, numrules)
    pvalold = partitionlogrank(partitionold, data)
    Xtest = Dtest.loc[:,~Dtest.columns.isin(['time','status'])]
    timetest = np.array(Dtest.time).reshape(-1,1)
    censtest = np.array(Dtest.status).reshape(-1,1)

    for itr in range(iterstart,iterstart + niter):
        if itr%20 == 0:
            print('Iter:{}'.format(itr))
    #         print(pvratio)
        # propose change to new rule list 
        testlist, qratio, ind = proposenewlist(rulelistold, Acopy, lenlist)
        if testlist == rulelistold:
            acceptancelist.append(0)
            Acopy = A.copy()
            #print('exit1')
            continue 
        testpart =  partitiondata(testlist, data, test = 0) # TODO change
        Acopy = Acopy + testpart['rulesnotused']
        testlist = testpart['rulelist']
        pvaltest = partitionlogrank(testpart, data)
        probtest = probrulelist2(testlist, meanrules, card_dict, numrules)
        pratio = probtest/probold
        if pvaltest/pvalold > 1e10:
            acceptancelist.append(1)
            Acopy = A.copy()
    #         print('x')
            continue 
        else: 
            try: 
                modtest, logliktest = buildmodel(testpart, data, modold, ind)
            except:
                Acopy = A.copy()
                continue
            lratio = np.exp(logliktest - loglikold)
            alpha = qratio*lratio*pratio
            if lratio == 'a':
                print('Iter:{}, lratio:{}, pratio:{}, qratio:{},alpha:{} | not accepted'.format(
                    itr,lratio,pratio,qratio,alpha))
                Acopy = A.copy()
                acceptancelist.append(2)
                continue
            else:
                if  0 == np.random.binomial(1,min(1,alpha)):
                    acceptancelist.append(3)
                    Acopy = A.copy()
                    continue
                else: 
                    print('Iter:{}, lratio:{}, pratio:{}, qratio:{},alpha:{},loglikelihood:{} | accepted'.format(
                        itr,lratio,pratio,qratio,alpha,logliktest))
                    loglikold = logliktest 
                    pvalold = pvaltest 
                    probold = probtest
                    rulelistold = testlist
                    lenlist = len(rulelistold)
                    modold = modtest
                    partitionold = testpart
                    p = predictive_densitygp_sampling(modold, Xtest, partitionold, timetest,
                                                      censtest, test = 1)
                    dlist.append((itr, rulelistold, loglikold, p[0]))
                    A = Acopy.copy()
                    acceptancelist.append(-1)
    res = {}
    res['rulelistold'] = rulelistold
    res['modold'] = modold
    res['loglikold'] = loglikold
    res['dlist'] = dlist
    res['acceptancelist']= acceptancelist
    
    return res


# build model from partition and data
def buildmodel(partition, data, modold = None, ind = 0):
    
    # partition: new partition
    # modold: older model
    # ind: index at which change has occured
    numparts = partition['numparts'] # new numparts
    m = {}
    for i in range(ind):
        m[i] = modold[i]
    for idx in range(ind, numparts):
        m[idx] = {}
        ruledata = data[partition[idx]['conditional']]
        mod = runmodel(ruledata)
        m[idx]['model'] = mod
        m[idx]['loglikelihood'] = mod.log_likelihood()
    loglikelihood = sum([m[i]['loglikelihood'] for i in range(len(m))])
    return m, loglikelihood

# run GPy model 
def runmodel(ruledata):
    
    time = np.array(ruledata.time).reshape((-1,1))
    cens = np.array(ruledata.status).reshape((-1,1))
    X = ruledata.loc[:,~ruledata.columns.isin(['time','status'])]
    Y_metadata = {'censored': 1-cens}
    
    lik = GPy.likelihoods.LogLogistic()
    lik.r_log_shape.set_prior(GPy.priors.Uniform(0,50))
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    
    kprior = GPy.kern.RBF(input_dim = X.shape[1], name = 'f_rbf', ARD = True)
    kprior.variance.set_prior(GPy.priors.Gamma.from_EV(1,10))
    kprior.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1,10))
    
    model = GPy.core.GP(X=X, Y=time, likelihood=lik,
                     inference_method=laplace_inf, kernel=kprior,
                     Y_metadata = Y_metadata)
    model.optimize('bfgs', messages=0, max_iters=1300)
    return model



# logrank test for screening criteria
def partitionlogrank(partition, data):

    ts = data[['time','status']]
    groups = [-1 for i in range(data.shape[0])]
    for idx in range(partition['numparts']):
        for i in range(data.shape[0]):
            if partition[idx]['conditional'][i] : groups[i] = idx
#     data['groups'] = groups
    results = multivariate_logrank_test(data['time'], groups, data['status'])
#     data.drop(['groups'], axis=1)
    return results.p_value


# plot survival curve
def plot_St(m, Xtest, num_samples=100, ax = None, label = None, conf_int = True):
    
    shape = m.LogLogistic.r_log_shape[0]
    
    inputs_mat = Xtest.copy()
    
    mu_f, covar_f = m._raw_predict(inputs_mat, full_cov=True)
    
    post_f_samples = np.random.multivariate_normal(mu_f.flatten(), covar_f[:,:], size=num_samples)

    post_scale_samples = np.exp(post_f_samples)
    
    t = np.linspace(0, m.Y.max(), 400)
    
    St = np.array([1.0 / (1 + (s / post_scale_samples)**shape) for s in t])
    
    mean_st = np.mean(St,axis = 2)
    
    ninety_five = np.percentile(mean_st, 95, axis=1)
    
    five = np.percentile(mean_st, 5, axis=1)
    
    mean = np.mean(mean_st, axis = 1)
    
    if ax is None:
        
        fig, ax = plt.subplots(1,1)
        
    if label is None:
        label = 'SGP (' + r'$ \beta $' + ' = '+str(round(shape,3))+') estimate'
    plt.title('Survival function')
    plt.plot(t, mean, label = label)
    if conf_int == True:
        plt.fill_between(t,five,ninety_five, alpha = 0.2)    
    plt.ylabel('$S(t)$')
    plt.xlabel('Time')
    plt.legend()

    return ax

def plotGPSRL_St(modold, partitionold,X):
    fig = plt.figure()
    ax = plot_St(modold[0]['model'], np.array(X[partitionold[0]['conditional']]),
                 label = '$\it{if}$' + ' ' + '$r_1$', ax = None, conf_int = True)
    for i in range(1,len(modold)-1):
        label = '$\it{else}$' +' '+ '$\it{if}$  ' + '$r_{%d}$'% (i+1)
        plot_St(modold[i]['model'], np.array(X[partitionold[i]['conditional']]),
                label = label, ax = ax, conf_int = True)
    label = '$\it{else}$'
    i=i+1
    plot_St(modold[i]['model'], np.array(X[partitionold[i]['conditional']]),
            label = label, ax = ax, conf_int = True)
    plt.tight_layout()
    return fig


# predictive density on test data 
def predictive_densitygp_sampling(gpmod, Xtest, partition, timetest, censtest,test = 1):
    
    orules = partition['rulelist'] 
    testpartition = partitiondata(orules, Xtest, test = test, minth = 5)
    testrules = testpartition['rulelist'] + ['otherwise']
    orules = orules + ['otherwise']
    testrulesnotused = testpartition['rulesnotused']
    if len(testrules) == 1:
        pass
    prediction_score = 0
    modidx = 0
    ruleidx = 0
    while True:
        if testrules[ruleidx] == orules[modidx]:
            model = gpmod[modidx]['model']
            X0 = Xtest[testpartition[ruleidx]['conditional']]
            #print(X0.shape)
            time0 = timetest[testpartition[ruleidx]['conditional']]
            cens0 = censtest[testpartition[ruleidx]['conditional']]
            pred = sum(model.log_predictive_density_sampling(np.array(
                X0),time0,Y_metadata = {'censored' : 1-cens0}))
            prediction_score += pred
                
            modidx  +=1
            ruleidx +=1
        else:
            modidx += 1 
        if ruleidx == testpartition['numparts']:
            break
            
    return prediction_score

def gp_concordance_index(m, Xtest, timetest, censtest, num_samples = 100):

#     m = m01
    shape = m.LogLogistic.r_log_shape[0]

    inputs_mat = np.array(Xtest.copy())

    mu_f, covar_f = m._raw_predict(inputs_mat, full_cov=True)

    post_f_samples = np.random.multivariate_normal(mu_f.flatten(), covar_f[:,:].squeeze(), size=num_samples)

    post_scale_samples = np.exp(post_f_samples)

    time1 = [[fisk.rvs(c = shape, loc=0, scale = a, size=1, random_state=None) for a in scale]
             for scale in post_scale_samples]
    c_indices = [concordance_index(t, timetest, censtest) for t in time1]

    return np.mean(c_indices)

def predictive_density_gpsrl(Xtest, rulelistold, modold, timetest, censtest):
    c = 0 
    for i in range(Xtest.shape[0]):
        xtest = Xtest.iloc[i:i+1,:]
        res = next((i for i,j in enumerate(rulelistold) if xtest.eval(j).bool()),len(rulelistold))
        c += modold[res]['model'].log_predictive_density_sampling(np.array(xtest),timetest[i].reshape(-1,1),
                              Y_metadata = {'censored':1-censtest[i].reshape(-1,1)})
    return(c)    


def gpsrl_predictive_density(modold, Xtest, rulelistold, timetest, censtest, num_samples = 100):
    k = Xtest.shape[0]
    score = 0
    for i in range(k): #range(Xtest.shape[0]):
        xtest = Xtest.iloc[i:i+1,:]
        res = next((i for i,j in enumerate(rulelistold) if xtest.eval(j).bool()),len(rulelistold))
        m = modold[res]['model']
        score+=m.log_predictive_density_sampling(np.array(
                xtest),timetest[i],Y_metadata = {'censored' : 1-censtest[i]})
    
    return score

def gpsrl_concordance_index(modold, Xtest, rulelistold, timetest, censtest, num_samples = 100):
    num_samples = 100;
    times = []
    k = Xtest.shape[0]
    for i in range(k): #range(Xtest.shape[0]):
        xtest = Xtest.iloc[i:i+1,:]
        res = next((i for i,j in enumerate(rulelistold) if xtest.eval(j).bool()),len(rulelistold))
        m = modold[res]['model']
        shape = m.LogLogistic.r_log_shape[0]

        inputs_mat = np.array(xtest.copy())

        mu_f, covar_f = m._raw_predict(inputs_mat, full_cov=True)

        post_f_samples = np.random.multivariate_normal(mu_f.flatten(), covar_f[:,:], size=num_samples)

        post_scale_samples = np.exp(post_f_samples)

        times.append([[fisk.rvs(c = shape, loc=0, scale = a, size=1, random_state=None) for a in scale]
             for scale in post_scale_samples])

    # times_arr = np.array(times).reshape(num_samples,Xtest.shape[0])
    times_arr = np.array(times).reshape(num_samples,k)
    time1 = [[times[i][j][0][0] for i in range(k)] for j in range(num_samples)]
    c_indices = [concordance_index(t, timetest[0:k], censtest[0:k]) for t in time1]
    
    return np.mean(c_indices)




## CODES for Rule Extraction from RandomSurvivalForest 

# decompose rules 
def decompose_rules(rule, nvar = 2):
    from itertools import combinations 
    newrules = [rule]
    newrules = newrules + rule.split('&')
    
    if nvar == 3: 
        newrules = [rule]
        combs =  rule.split('&')
        newrules = newrules + [combs[2]]+ [combs[1]+'&'+combs[2]] + [combs[0]+'&'+combs[1]]
        
    return newrules

# get rules from rsf 
def get_rules_rsf(rsf, D):
    
    rules = []
    for tree_idx, est in enumerate(rsf.estimators_):
        rules = rules + rule_extract_tree(est,D)
        
    return rules

# extract rules from trees
def rule_extract_tree(clf, D):
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    def find_path(node_numb, path, x):
            path.append(node_numb)
            if node_numb == x:
                return True
            left = False
            right = False
            if (children_left[node_numb] !=-1):
                left = find_path(children_left[node_numb], path, x)
            if (children_right[node_numb] !=-1):
                right = find_path(children_right[node_numb], path, x)
            if left or right :
                return True
            path.remove(node_numb)
            return False


    def get_rule(path, column_names):
        mask = ''
        for index, node in enumerate(path):
            #We check if we are not in the leaf
            if index!=len(path)-1:
                # Do we go under or over the threshold ?
                if (children_left[node] == path[index+1]):
                    mask += "({} <= {}) \t".format(column_names[feature[node]], np.round(threshold[node],2))
                else:
                    mask += "({} > {}) \t".format(column_names[feature[node]], np.round(threshold[node],))
        # We insert the & at the right places
        mask = mask.replace("\t", "& ", mask.count("\t") - 1)
        mask = mask.replace("\t", "")
        return mask

    leave_id = clf.tree_.apply(np.array(D.iloc[:,2:D.shape[1]], dtype = 'float32'))

    paths ={}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))

    rules = {}
    for key in paths:
        rules[key] = get_rule(paths[key], D.columns[2:len(D.columns)])

    return list(rules.values())