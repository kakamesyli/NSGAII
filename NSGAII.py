#!usr/bin/env python
'''
Created on 202101011

@author: think
'''

'''
Created on 20210111
@author: kakamesyli
'''
    
class Pop(object):
    def __init__(self,pop_index):
        self.index = pop_index; #个体编号
        self.var = []; #个体变量值
        self.output = []; #个体输出值
        self.pareto_rank = 0;
        self.crowding = 0;
        self.dom_num = 0;
        self.dom_asm = [];
        
class Chromo(object):
    def __init__(self,pop_num):
        self.pop = [];
        self.rank = 0;
        self.rank_asm = [];#某等级下的种群序号集合
        for i in range(pop_num):
            self.pop.append(Pop(i));
        
def non_dominate_sort(chormo, f_num):
    for i in chormo:
        chromo.dom_num = 0;
        chromo.dom_asm = [];
        for j in chormo:
            less = 0;
            equal = 0;
            greater = 0;
            if (i.value[0] < j.value[0] and i.value[1] < j.value[1]):
                less = less + 1;
            elif (i.value[0] == j.value[0] and i.value[1] == j.value[1]):
                equal = equal + 1;
            else:
                greater = greater + 1;
            if (less == 0 and equal != f_num):
                chormo.dom_num = chormo.dom_num +1;
            elif (greater == 0 and equal != f_num):
                chormo.dom_asm.append(j);
        if chormo.dom_num == 0:
            chormo.pareto_rank = 1;
            chormo.rank_asm.append(i);
    while (len(chormo.rank_asm) != 0):
        for pop in chormo.rank_asm:
            if (len(chormo.dom_asm) != 0):
                