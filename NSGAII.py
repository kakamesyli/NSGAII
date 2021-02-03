#!usr/bin/env python
'''
Created on 202101011
@author: think
'''
# from numpy.lib.user_array import temp
# from numpy.random.mtrand import pareto
# from _ctypes_test import func

'''
Created on 20210111
@author: kakamesyli
'''
import random
import operator


class Pop(object):

    def __init__(self, pop_index, x_min, x_max, x_num, f_num, fun_name):
        self.index = pop_index;  # #code
        self.var = [];  #
        for i in range(x_num):
            self.cal_var(x_min, x_max);
        self.value = [];  #
        for j in range(f_num):
            self.cal_output(self.var, x_num, fun_name);
        self.pareto_rank = 0;
        self.crowding = 0;
        self.dom_num = 0;
        self.dom_asm = [];

    def cal_var(self, x_min, x_max):
        self.var.append(x_min + (x_max - x_min) * random.random());
    
    def cal_output(self, x, x_num, fun_name):
        self.value.append(obj_fun(x, x_num, fun_name));
    
    def create_rank_asm(self, Pop):
        self.rank_asm.append(Pop);

        
class Chromo(object):

    def __init__(self, pop_num, x_min, x_max, x_num, f_num, fun_name):
        self.pop_num = pop_num;
        self.pop_asm = [];
        self.rank_asm = []; 
        for i in range(self.pop_num):
            self.create_pop_asm(i, x_min, x_max, x_num, f_num, fun_name);
    
    def create_pop_asm(self, pop_index, x_min, x_max, x_num, f_num, fun_name):
        self.pop_asm.append(Pop(pop_index, x_min, x_max, x_num, f_num, fun_name));
        
    def non_dominate_sort(self, pop_asm, f_num):
        pareto_rank = 1;
        for pop_i in pop_asm:
            for pop_j in pop_asm:
                less = 0;
                equal = 0;
                greater = 0;
                for k in range(f_num):
                    if (pop_i.value[k] < pop_j.value[k]):
                        less = less + 1;
                    elif (pop_i.value[k] == pop_j.value[k]):
                        equal = equal + 1;
                    else:
                        greater = greater + 1;
                if (less == 0 and equal != f_num):
                    pop_i.dom_num = pop_i.dom_num + 1;
                elif (greater == 0 and equal != f_num):
                    pop_i.dom_asm.append(pop_j);
                if pop_i.dom_num == 0:
                    pop_i.pareto_rank = pareto_rank;
                    self.rank_asm[pareto_rank].append(pop_i);
        while (len(self.rank_asm[pareto_rank]) != 0):
            for rank_ele in self.rank_asm[pareto_rank]:
                if (len(rank_ele.dom_asm) != 0):
                    for dom_asm_ele in rank_ele.dom_asm:
                        dom_asm_ele.dom_num = dom_asm_ele.dom_num - 1;
                        if dom_asm_ele.dom_num == 0:
                            dom_asm_ele.pareto_rank = pareto_rank + 1;
                            self.rank_asm[pareto_rank + 1].append(dom_asm_ele);
            pareto_rank = pareto_rank + 1;
            
    def crowding_distance_sort(self, pop_asm, f_num):
        # temp = pop_asm.sorted(key=lambda x:x.pareto_rank, reverse=False);
        for pareto_rank in range(len(self.rank_asm)):
            for func in range(f_num):
                func_temp = []
                func_temp[pareto_rank] = sorted(self.rank_asm[pareto_rank], key=lambda x:x.value[func], reverse=False)
                current_index = 0;
                fmin = func_temp[pareto_rank][0].value[func]
                fmax = func_temp[pareto_rank][-1].value[func]
                func_temp[pareto_rank][0] = float('Inf')
                func_temp[pareto_rank][-1] = float('Inf')
                for i in range(1, len(func_temp[pareto_rank]) - 2):  # for pop in func_temp[pareto_rank][1, -2]:
                    next_fun_value = func_temp[pareto_rank][i - 1].value[func]  # next_fun_value = func_temp[pareto_rank][pop.index()-1].value[func]
                    pre_fun_value = func_temp[pareto_rank][i + 1].value[func]  # pre_fun_value = func_temp[pareto_rank][pop.index()+1].value[func]
                    if fmax == fmin:
                        func_temp[pareto_rank][i].crowding = float('Inf')
                    else:
                        func_temp[pareto_rank][i].crowding = func_temp[pareto_rank][i].crowding + (next_fun_value - pre_fun_value) / (fmax - fmin)
            func_temp[pareto_rank] = sorted(func_temp[pareto_rank], key=lambda x:x.index, reverse=False)
            for i in range(len(func_temp[pareto_rank])):
                self.rank_asm[pareto_rank][i].crowding = func_temp[pareto_rank][i].crowding
                
                '''
            for pop in temp[current_index, len(self.rank_asm[pareto_rank]) - 1]:
                for func in range(f_num):
                    fun_value_temp[func] = 
                    fun_value_temp[func].append(pop.value[func])
            
                y_asm.append(pop)
            for f in range(f_num):
                f_sort[f] !! = y_asm.sorted(key=lambda x:x.value[f], reverse=False)
                fmin = f_sort[1];
                fmax = f_sort[-1]
                f_sort[1].crowding = float('inf')
                f_sort[-1].crowding = float('inf')
                for f_s in range(2, len(f_sort) - 1):
                    pre_f = f_sort[f_s - 1]
                    next_f = f_sort[f_s + 1]
                    if fmin == fmax:
                        f_sort[f_s] = float('inf')
                    else:
                        f_sort[f_s] = (next_f - pre_f) / (fmax - fmin)
            for f in range(f_num):
                nd = 
'''


def non_dominate_sort(pop_asm, f_num):
    rank_asm = []
    rank_asm_temp = []
    pareto_rank = 1
    flag = 0
    for pop_i in pop_asm:
        for pop_j in pop_asm:
            less = 0;
            equal = 0;
            greater = 0;
            for k in range(f_num):
                if (pop_i.value[k] < pop_j.value[k]):
                    less = less + 1
                elif (pop_i.value[k] == pop_j.value[k]):
                    equal = equal + 1
                else:
                    greater = greater + 1
            if (less == 0 and equal != f_num):
                pop_i.dom_num = pop_i.dom_num + 1
            elif (greater == 0 and equal != f_num):
                pop_i.dom_asm.append(pop_j)
        if pop_i.dom_num == 0:
                pop_i.pareto_rank = pareto_rank
                rank_asm_temp.append(pop_i)
                rank_asm.append(rank_asm_temp)
    while (len(rank_asm[pareto_rank-1]) != 0 and flag == 0):
        rank_asm_temp = []
        for rank_ele in rank_asm[pareto_rank-1]:
            if (len(rank_ele.dom_asm) != 0):
                for dom_asm_ele in rank_ele.dom_asm:
                    dom_asm_ele.dom_num = dom_asm_ele.dom_num - 1
                    if dom_asm_ele.dom_num == 0:
                        dom_asm_ele.pareto_rank = pareto_rank + 1
                        rank_asm_temp.append(dom_asm_ele)
                rank_asm.append(rank_asm_temp)
                pareto_rank = pareto_rank + 1
            else:
                flag = 1
        
    return rank_asm
def crowding_distance_sort(rank_asm, f_num):
    # temp = pop_asm.sorted(key=lambda x:x.pareto_rank, reverse=False);
    for pareto_rank in range(len(rank_asm)):
        for func in range(f_num):
            func_temp = []
            rank_asm_crowding = []
            func_temp.append(sorted(rank_asm[pareto_rank], key=lambda x:x.value[func], reverse=False))
            current_index = 0
            fmin = func_temp[pareto_rank][0].value[func]
            fmax = func_temp[pareto_rank][-1].value[func]
            func_temp[pareto_rank][0].crowding = float('Inf')
            func_temp[pareto_rank][-1].crowding = float('Inf')
            for i in range(1, len(func_temp[pareto_rank]) - 2):  # for pop in func_temp[pareto_rank][1, -2]:
                next_fun_value = func_temp[pareto_rank][i - 1].value[func]  # next_fun_value = func_temp[pareto_rank][pop.index()-1].value[func]
                pre_fun_value = func_temp[pareto_rank][i + 1].value[func]  # pre_fun_value = func_temp[pareto_rank][pop.index()+1].value[func]
                if fmax == fmin:
                    func_temp[pareto_rank][i].crowding = float('Inf')
                else:
                    func_temp[pareto_rank][i].crowding = func_temp[pareto_rank][i].crowding + (next_fun_value - pre_fun_value) / (fmax - fmin)
        func_temp[pareto_rank] = sorted(func_temp[pareto_rank], key=lambda x:x.index, reverse=False)
        for i in range(len(func_temp[pareto_rank])):
            #rank_asm_crowding[pareto_rank][i].crowding = func_temp[pareto_rank][i].crowding
            rank_asm[pareto_rank][i].crowding = func_temp[pareto_rank][i].crowding
    return rank_asm_crowding
        
def elitsm(pop_num, chromo2):
    chromo_temp = sorted(chromo2, key=lambda x:x.pareto_rank, reverse=False)
    pre_ind = 0
    current_rank = 1
    current_ind = 1
    ind = []
    current_rank_temp = []
    chromo_elit = []
    for pop in chromo_temp:       
        if pop.pareto_rank == current_rank:
            current_ind = current_ind + 1
        elif pop.pareto_rank > current_rank:
            ind[current_rank] = current_ind
            current_rank = current_rank + 1
            current_ind = current_ind + 1
        ind[current_rank] = current_ind
            
    for i in range(len(ind)):
        if ind[i] < pop_num or ind[i] == pop_num:
            chromo_elit.append(chromo_temp[i - 1])
        elif ind[i] > pop_num:
            for j in range(ind[i - 1], pop_num - ind[i - 1]):
                chromo_elit.append(chromo_temp[j - 1])

            '''
            current_rank_temp[current_rank].append(pop)
            if current_ind < pop_num and current_ind == pop_num:
                chromo_elit.append(current_rank_temp[current_rank])
            elif current_ind > pop_num:
                crowding_distance_sort(current_rank_temp[current_rank])
        elif pop.pareto_rank > current_rank:
            current_rank = current_rank + 1
            current_ind = current_ind + 1
            current_rank_temp[current_rank].append(pop)
        else:
            print('error')
        if pop.pareto_rank == current_rank:
            if current_ind < pop_num:
                chromo_elit.append(pop)
                current_ind = current_ind + 1
            elif current_ind > pop_num and current_ind == pop_num:
                
        elif pop.pareto_rank > current_rank:
            
        
            
            chromo_temp = chromo_temp + 1
        elif current_ind > pop_num and current_ind == pop_num:
            
    for i in range(1,max(chromo_temp.pareto_rank)):
        current_index_pareto = 
        '''
    return chromo_elit

            
def obj_fun(x, x_num, fun_name):
    if operator.eq(fun_name, 'ZDT1'):
        f = []
        f.append(x[0])
        s = 0
        for i in range(1,len(x)):
            s = s + x[i]
        g = 1 + 9 * (s / (x_num - 1))
        f.append(g * (1 - (f[0] / g) ** 0.5))
        return f;

    
class Person():

    def __init__(self, age, name):
        self.age = age
        self.name = name


def personSort():
    persons = [Person(age, name) for (age, name) in [(12, "lili"), (18, "lulu"), (16, "kaka"), (12, "xixi")]]
    persons.sort(key=lambda x:x.name, reverse=False);
    for element in persons:
        print (element.age, ":", element.name);


if __name__ == "__main__":
    x_min = 0
    x_max = 100
    x_num = 100
    f_num = 1
    fun_name = 'ZDT1'
    pop_num = 5
    chromo = Chromo(pop_num, x_min, x_max, x_num, f_num, fun_name)
    chromo_domi = non_dominate_sort(chromo.pop_asm, f_num)
    chromo_domi_crowding = crowding_distance_sort(chromo_domi, f_num)
    