#!usr/bin/env python
'''
Created on 202101011
@author: think
'''

'''
Created on 20210111
@author: kakamesyli
'''
import random
import operator
class Pop(object):
    def __init__(self,pop_index,x_min,x_max,x_num,f_num,fun_name):
        self.index = pop_index; #个体编号
        self.var = []; #个体变量值
        for i in range(x_num):
            self.cal_var(x_min,x_max);
        self.value = []; #个体输出值
        for j in range(f_num):
            self.cal_output(self.var,x_num,fun_name[j]);
        self.pareto_rank = 0;
        self.crowding = 0;
        self.dom_num = 0;
        self.dom_asm = [];

    def cal_var(self,x_min,x_max):
        self.var.append(x_min + (x_max-x_min) * random.random());
    
    def cal_output(self,x,x_num,fun_name):
        self.output.append(obj_fun(x,x_num,fun_name));
    
    def create_rank_asm(self,Pop):
        self.rank_asm.append(Pop);
        
class Chromo(object):
    def __init__(self,pop_num,x_min,x_max,x_num,f_num,fun_name):
        self.pop_num = pop_num;
        self.pop_asm = [];
        self.rank_asm = []; 
        
        '''
        定义个体集合
        '''
        for i in range(self.pop_num):
            self.create_pop_asm(self,i,x_min,x_max,x_num,f_num,fun_name);
    
    def create_pop_asm(self,pop_index,x_min,x_max,x_num,f_num,fun_name):
            self.pop_asm.append(Pop(pop_index,x_min,x_max,x_num,f_num,fun_name));
    
        
    def non_dominate_sort(self,pop_asm,f_num):
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
                    pop_i.dom_num = pop_i.dom_num +1;
                elif (greater == 0 and equal != f_num):
                    pop_i.dom_asm.append(pop_j);
                if pop_i.dom_num == 0:
                    pop_i.pareto_rank = pareto_rank;
                    self.rank_asm.append((pop_i.pareto_rank,pop_i));
        while (len(self.rank_asm) != 0):
            for rank_ele in self.rank_asm:
                if (len(rank_ele.dom_asm) != 0):
                    for dom_asm_ele in rank_ele.dom_asm:
                        dom_asm_ele.dom_num = dom_asm_ele.dom_num - 1;
                        if dom_asm_ele.dom_num == 0:
                            dom_asm_ele.pareto_rank = pareto_rank + 1;
                            self.rank_asm.append((dom_asm_ele.pareto_rank,dom_asm_ele));
            pareto_rank = pareto_rank + 1;
    def crowding_distance_sort(self):
        temp = self.pop_asm.sort(cmp=None, key=None, reverse=False);
        current_dist = 0;
        
def obj_fun(x,x_num,fun_name):
    if operator.eq(fun_name, 'ZDT1'):
        f = [];
        f[0] = x[1];
        s = 0;
        for i in range(x):
            s = s + x[i];
            g = 1 + 9 * (s / (x_num - 1));
            f[1] = g * (1 - (f[1] / g) ^ 0.5);
        return f;
