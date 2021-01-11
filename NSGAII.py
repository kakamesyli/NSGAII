#!usr/bin/env python
'''
Created on 20210111

@author: kakamesyli
'''


class chromo(object):

    def __init__(self):
        self.index = 0;
        self.pop = [];  # 种群点
        self.value = [];  # 种群输出值
        self.pareto_rank = 0;
        self.crowding = 0;
        self.dom_num = 0;
        self.dom_asm = [];
        self.rank_asm = [];#某等级下的种群序号集合
    
        
def non_dominate_sort(chormo, f_num):
    for pop_i in chormo.pop:
        chromo.dom_num = 0;
        chromo.dom_asm = [];
        for pop_j in chormo.pop:
            less = 0;
            equal = 0;
            greater = 0;
            if (pop_i.value[0] < pop_j.value[0] and pop_i.value[1] < pop_j.value[1]):
                less = less + 1;
            elif (pop_i.value[0] == pop_j.value[0] and pop_i.value[1] == pop_j.value[1]):
                equal = equal + 1;
            else:
                greater = greater + 1;
            if (less == 0 and equal != f_num):
                chormo.dom_num = chormo.dom_num +1;
            elif (greater == 0 and equal != f_num):
                chormo.dom_asm.append(pop_j);
        if chormo.dom_num == 0:
            chormo.pareto_rank = 1;
            chormo.rank_asm.append(pop_i);
    while (len(chormo.rank_asm) != 0):
        for pop in chormo.rank_asm:
            if (len(chormo.dom_asm) != 0):
                
