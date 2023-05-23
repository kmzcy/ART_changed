import torch
import torch.nn as nn
import numpy as np
from Temp_move import T_change

class myART(nn.Module):
    def __init__(self, input_size, class_num, rou = 0.83, beta = 0.01, alpha = 0.05, linera_layer_num = 0, device='cuda:0', layer_num = 3):
        """
        初始化
        input:
            input_size: 图片向量的长度, 模板向量的长度的1/2
            rou: 警戒值(阈值, 0到1, 由0开始向1不断增加, 阈值越小, 一个模板可以匹配到的数目越多, 可以形象的表述为圆越大)
            beta: 层内ART模板学习率(当判定向量I属于模板M时, 模型会使用向量I对该模板进行更新)
            alpha: 计算匹配值时分母上的超参数,防止分母为0
            linera_layer_num: 使用的哪一层的线性层
            device: 显卡编号
            layer_num: ART模型的层数
        """
        super(myART, self).__init__()

        self.rou = [] # 警戒值
        for i in range(layer_num):
            print(i, " layer the rou is: ", rou + 0.01 * i)
            self.rou.append(rou + 0.01 * i)

        self.linera_layer_num = linera_layer_num # 使用的哪一层的线性层
        self.beta = beta # 层内ART模板学习率(保留多少(1 - beta)旧模板的成分)
        self.alpha = alpha # 层内ART相似度计算参数
        self.ilen = input_size # 展平向量的长度，模板向量的长度的1/2
        self.layer_num = layer_num # ART的层数
        self.class_num = class_num # 总样本类别数
        # 待取舍策略
        self.label_templates = [] # 在取平均的策略下用于存放取平均后的最终模板
        
        self.isclean = [] # -1表示模板不纯，其他数字表示标签号，初始化为-2
        for i in range(layer_num):
            self.isclean.append([])

        self.lab_count = [] # 存储每个模板下每种标签有几个样本,字典格式存储
        for i in range(layer_num):
            self.lab_count.append([])

        self.father_tmp = [] # 存储该模板来自于上层的第几个模板，无源模板存储为-1
        for i in range(layer_num):
            self.father_tmp.append([])

        self.ancestor = [] # 存储最开始的模板，无源模板存储为-1
        for i in range(layer_num):
            self.ancestor.append([])

        self.template = []  # 存储模板
        for i in range(layer_num):
            self.template.append([])

        self.isNew = [] # 用于标识模板是否在本轮中新创建1为新，可参与更新，0为旧不可参与更新
        for i in range(layer_num):
            self.isNew.append([])
        
        self.isCount = [] # 用于关模版计数的更新
        for i in range(layer_num):
            self.isCount.append([])

        self.liners = [] # 存储层间映射函数
        for i in range(layer_num - 1):
            self.liners.append({})

        self.target_label= [] # 记录每个模板的目标标签（纯净后属于哪一类），无源模板该值为-1
        for i in range(layer_num):
            self.target_label.append([])

        self.device = torch.device(device if torch.cuda.is_available() else "cpu") 
        
    def change_layer(self, linera_layer_num = 0):
        self.linera_layer_num = linera_layer_num # 改变线性层

    def change_rou(self, rou):
        """
        改变阈值, 不同的训练阶段可以对阈值进行调整
        """
        self.rou = rou

    def change_beta(self, beta):
        """
        改变更新模板时的更新量，保留多少(1 - beta)旧模板的成分
        """
        self.beta = beta

    def save_all(self,save_path):
        """
        用一个字典将所有的值都存储起来。

        input:
            save_path:存储路径
        """
        all = {} # 字典
        all["isclean"] = self.isclean # 模板是否纯净
        all["lab_count"] = self.lab_count # 存储每个模板下每种标签有几个样本,本身就是个字典
        all["father_tmp"] = self.father_tmp # 模板的父模板
        all["ancestor"] = self.ancestor # 模板的祖先
        all["template"] = self.template  # 存储模板本身
        all["isNew"] = self.isNew # 是不是新模板
        all["isCount"] = self.isCount # 对不对这个模板重新计数
        all["liners"] = self.liners # 线性层
        all["target_lab"] = self.target_label # 记录每个模板的目标标签（纯净后属于哪一类），无源模板该值为-1
        all["rou"] = self.rou
           
        torch.save(all, save_path)

    def load_all(self,save_path):
        """
        将所有的值读入到模型的参数中。

        input:
            save_path:存储路径
        """
        # all = torch.load(save_path, map_location={'cuda:0': 'cuda:1'})
        all = torch.load(save_path)
        
        if len(self.isclean) >= len(all["isclean"]):
            isclean_lenth = len(all["isclean"])
        else:
            isclean_lenth = len(self.isclean)
            
        for i in range(isclean_lenth):
            self.isclean[i] = all["isclean"][i]
            
        if len(self.lab_count) >= len(all["lab_count"]):
            lab_count_lenth = len(all["lab_count"])
        else:
            lab_count_lenth = len(self.lab_count)

        for i in range(lab_count_lenth): 
            self.lab_count[i] = all["lab_count"][i]
            
        if len(self.father_tmp) >= len(all["father_tmp"]):
            father_tmp_lenth = len(all["father_tmp"])
        else:
            father_tmp_lenth = len(self.father_tmp)

        for i in range(father_tmp_lenth):
            self.father_tmp[i] = all["father_tmp"][i]
            
        if len(self.ancestor) >= len(all["ancestor"]):
            ancestor_lenth = len(all["ancestor"])
        else:
            ancestor_lenth = len(self.ancestor)            
        
        for i in range(ancestor_lenth):
            self.ancestor[i] = all["ancestor"][i]

        if len(self.template) >= len(all["template"]):
            template_lenth = len(all["template"])
        else:
            template_lenth = len(self.template)             
            
        for i in range(template_lenth):
            self.template[i] = all["template"][i]

        # 将template放入对应的设备
        for i in range(len(self.template)):
            for j in range(len(self.template[i])):
                if self.template[i][j].device != self.device:
                        self.template[i][j] = self.template[i][j].to(self.device)

        if len(self.isNew) >= len(all["isNew"]):
            isNew_lenth = len(all["isNew"])
        else:
            isNew_lenth = len(self.isNew)                     
                    
        for i in range(isNew_lenth):
            self.isNew[i] = all["isNew"][i]
        
        if len(self.isCount) >= len(all["isCount"]):
            isCount_lenth = len(all["isCount"])
        else:
            isCount_lenth = len(self.isCount)           
        
        for i in range(isCount_lenth):
            self.isCount[i] = all["isCount"][i]
            
        if len(self.liners) >= len(all["liners"]):
            liners_lenth = len(all["liners"])
        else:
            liners_lenth = len(self.liners)              
            
        for i in range(liners_lenth):
            self.liners[i] = all["liners"][i]
            
        if len(self.target_label) >= len(all["target_lab"]):
            target_label_lenth = len(all["target_lab"])
        else:
            target_label_lenth = len(self.target_label)                   
            
        for i in range(target_label_lenth):
            self.target_label[i] = all["target_lab"][i]

        if len(self.rou) >= len(all["rou"]):
            rou_lenth = len(all["rou"])
        else:
            rou_lenth = len(self.rou)                   
            
        for i in range(rou_lenth):
            self.rou[i] = all["rou"][i]

    def distance(self, ms, input, labels, layer):# 增强正样本，用正常的增强方式
        """
        计算并返回损失函数需要的值 ms是预测模板编号序列 用于最终的损失函数。

        input:
            ms: 是预测到的模板的编号序列, 序列中的每个模板都会和输入的样本求一遍余弦相似度
            input: 输入样本 
            labele: 输入样本对应的标签
            layer: 层数
        return:
            temp_num: 在本层需要更新损失的模板编号
            Ts: 余弦相似度
            labels_after: label
        """
        # 将样本向量映射到0到1之间
        mapping = nn.Sigmoid()
        input = mapping(input)

        # 使用补码编码扩展向量
        input = self.F0(input)

        temp_num = []
        labels_after = []
        distance_all = []
        dis_with_other_all = []
        samility_all = []
        resonance_all = []

        for i in range(len(ms)):
            if ms[i] < 0: # -3标识不属于任何模板
                    distance = torch.tensor(0)
                    dis_with_other = torch.tensor(0)
                    samility = 0
                    resonance = 0

                    temp_num.append(-3)
                    distance_all.append(distance)
                    dis_with_other_all.append(dis_with_other)
                    labels_after.append(labels[i])
                    samility_all.append(samility)
                    resonance_all.append(resonance)
            else:
                for j in range(len(self.template[layer])):
                    if ms[i] == self.father_tmp[layer][j]: # 样本匹配到的上一层的模板如果是一个本层模板的父模板，那么这个样本就归属这个本层模板
                        x = input[i].float() # x是样本
                        y = self.template[layer][j].float() # y是匹配到的模板

                        # 计算样本与本模板的距离            
                        distance = torch.sum(torch.pow(torch.abs(torch.add(x, - y)), 2))

                        # 计算本模板与其他模板的距离 
                        dis_with_other = 0
                        for k in range(len(self.template[layer])):     
                            if k != j:
                                dis_with_other += torch.sum(torch.pow(torch.abs(torch.add(self.template[layer][k].detach(), - y)), 2))

                        dis_with_other = dis_with_other/len(self.template[layer]) # 取平均

                        # 计算与本模板的相似度
                        and_L1 = torch.norm(self.fuzzy_and(x, y), p = 1)
                        tmp_L1 = torch.norm(y, p = 1)
                        samility = and_L1/(self.alpha+tmp_L1)

                        # 计算与本模板的警戒值
                        and_L2 = torch.norm(self.fuzzy_and(x, y), p = 1)
                        xi_L2 = torch.norm(x, p = 1)
                        resonance = and_L2/xi_L2

                        temp_num.append(j)
                        distance_all.append(distance)
                        dis_with_other_all.append(dis_with_other)
                        labels_after.append(labels[i])
                        samility_all.append(samility)
                        resonance_all.append(resonance)
            
        # 模板temp_num匹配到了一个真实标签为labels_after的样本，这个样本与本模板的距离，本模板与其他模板的距离, 和样本与本模板的相似度，共振值
        return temp_num, distance_all, dis_with_other_all, labels_after, samility_all, resonance_all
    
    def distance_pure(self, ms, input, layer):
        """
        输入样本和匹配到的模板，返回样本和该模板的相似度、共振值、距离、样本和其他模板的平均距离
        和上面那个distance不同的是, distence传入的是上一层的模板。匹配的是，以改模板为父模板的模板，这个方法则是传入的是模板本身
        input:
            ms: 是预测到的模板的编号序列, 序列中的每个模板都会和输入的样本求一遍余弦相似度
            input: 输入样本 
            layer: 层数
        return:
            
        """
        # 将样本向量映射到0到1之间
        mapping = nn.Sigmoid()
        input = mapping(input)

        # 使用补码编码扩展向量
        input = self.F0(input)

        distance_all = []
        dis_with_other_all = []
        samility_all = []
        sam_with_other_all = []
        resonance_all = []
        res_with_other_all = []

        for i in range(len(ms)):
            x = input[i].float() # x是样本

            if ms[i] >= 0:
                # 计算样本与本模板的距离 
                y = self.template[layer][ms[i]].float() # y是匹配到的模板
                print("In distance_pure the template matched y: ", y) 
                distance = torch.sum(torch.pow(torch.abs(torch.add(x, - y)), 2))

                # 计算样本与其他模板的距离 
                dis_with_other = 0
                for k in range(len(self.template[layer])): 
                    if k != ms[i]: 	
                        dis_with_other += torch.sum(torch.pow(torch.abs(torch.add(self.template[layer][k].detach(), - x)), 2))

                dis_with_other = dis_with_other/len(self.template[layer]) # 取平均

                # 计算样本与本模板的相似度
                and_L1 = torch.norm(self.fuzzy_and(x, y), p = 1)
                tmp_L1 = torch.norm(y, p = 1)
                samility = and_L1/(self.alpha + tmp_L1)

                # 与样本其他模板的相似度
                sam_with_other = 0
                for k in range(len(self.template[layer])): 
                    if k != ms[i]:   
                        and_L1 = torch.norm(self.fuzzy_and(x, self.template[layer][k]), p = 1)
                        tmp_L1 = torch.norm(self.template[layer][k], p = 1)
                        samility_mid = and_L1/(self.alpha + tmp_L1)
                        sam_with_other += samility_mid

                sam_with_other = sam_with_other/len(self.template[layer]) # 取平均

                # 计算样本与本模板的警戒值
                and_L2 = torch.norm(self.fuzzy_and(x, y), p = 1)
                xi_L2 = torch.norm(x, p = 1)
                resonance = and_L2/xi_L2
                
                # 计算样本与其他模板的警戒值, 样本和其他模板的警戒值取平均，外面再对每个样本对所有模板的平均警戒值求平均
                res_with_other = 0
                for k in range(len(self.template[layer])): 
                    if k != ms[i]:   
                        and_L2 = torch.norm(self.fuzzy_and(x, self.template[layer][k]), p = 1)
                        xi_L2 = torch.norm(x, p = 1)
                        resonance_mid = and_L2/xi_L2
                        res_with_other += resonance_mid

                res_with_other = res_with_other/len(self.template[layer]) # 取平均

                distance_all.append(distance)
                dis_with_other_all.append(dis_with_other)
                samility_all.append(samility)
                sam_with_other_all.append(sam_with_other)
                resonance_all.append(resonance)
                res_with_other_all.append(res_with_other)
            else:
                distance = torch.tensor(0)
                dis_with_other = torch.tensor(0)
                samility = 0
                sam_with_other = 0
                resonance = 0
                res_with_other = 0

                distance_all.append(distance)
                dis_with_other_all.append(dis_with_other)
                samility_all.append(samility)
                sam_with_other_all.append(sam_with_other)
                resonance_all.append(resonance)
                res_with_other_all.append(res_with_other)

        return distance_all, dis_with_other_all, samility_all, sam_with_other_all, resonance_all, res_with_other_all

    def fuzzy_and(self, a, b):
        """
        对a,b向量做模糊与,输出张量。对于a和b之中的每个值取最小值

        input:
            a: 向量a
            b: 向量b 
        return:
            torch.tensor(out_vec): a与b的模糊与
        """
        if len(a) != len(b): # 模糊与必须两向量长度相同
            print("fuzzy_and fault! "+ str(len(a)) +" "+ str(len(b)))
            exit(0)

        return torch.tensor(a - torch.relu(a - b))

    def update_t(self, xi, layer, j):
        """
        更新模板

        input:
            xi: 当前样本向量
            layer: 当前层编号 
            j:第几个模板
        """
        xi_tmp = self.fuzzy_and(xi, self.template[layer][j])
        xi_tmp = xi_tmp.to(self.device)
        self.template[layer][j] = self.beta*xi_tmp + (1-self.beta)*self.template[layer][j]   

    def create_t(self, xi, layer, father, traget_label=-1): 
        """
        创建新模板

        input:
            xi: 当前样本向量
            layer: 当前层编号 
            father: 父模板
            traget_label:由于该模板中仅有这一个类别，因此直接设为不纯-1,具体情况统计标签时会更新
        """
        
        self.template[layer].append(xi) # 直接把这个样本作为初始模板
        print("create template: ", len(self.template[layer]) - 1, " layer: ", layer)
        self.father_tmp[layer].append(father)

        if layer == 0:
            self.ancestor[layer].append(-1)
        elif layer == 1:
            self.ancestor[layer].append(father)
        else:
            self.ancestor[layer].append(self.ancestor[layer-1][father])

        self.lab_count[layer].append({})
        self.target_label[layer].append(traget_label)
        self.isclean[layer].append(-2) # 初始化 -2 代表新初始化的模板
        self.isNew[layer].append(1) # 用于标识是否为本层新创建的模板 1为是0为否
        self.isCount[layer].append(1) # 用于标识是否为需要统计标签的模板 1为是0为否
        
    def delete_empty(self):
        """
        删除所有空模板
        """
        to_delete = []
        for i in range(len(self.lab_count)):
            for j in range(len(self.lab_count[i])):
                if not bool(self.lab_count[i][j]):
                    to_delete.append((i, j))

        for i in range(len(to_delete)):     
            del self.template[to_delete[i][0]][to_delete[i][1]] # 直接把这个样本作为初始模板
            del self.father_tmp[to_delete[i][0]][to_delete[i][1]]
            del self.lab_count[to_delete[i][0]][to_delete[i][1]]   
            del self.target_label[to_delete[i][0]][to_delete[i][1]]
            del self.isclean[to_delete[i][0]][to_delete[i][1]]
            del self.ancestor[to_delete[i][0]][to_delete[i][1]]
            del self.isNew[to_delete[i][0]][to_delete[i][1]]
            del self.isCount[to_delete[i][0]][to_delete[i][1]]

            # if to_delete[i][0] > 0:
                # del self.liners[to_delete[i][0] - 1][to_delete[i][1]]
                
            for j in range(i, len(to_delete)):
                to_delete[j] =(to_delete[j][0], to_delete[j][1] - 1)

    def update_label(self, xi, layer, j, label): 
        """
        更新j模板的标签状况, 加入新的标签。
        若代表标签的字典中没有标签, 加入这个标签, 在记录是否纯净的数组上标这个标签代表纯净。
        如果这个字典中有这个标签, 计数加1, 纯净标识不变
        如果这个字典中没有这个标签，且字典还有其他标签, 加入这个标签, 并标为不纯净
        input:
            xi: 当前样本向量
            layer: 当前层编号 
            j: 要更新的模板编号
            label: xi的标签
        """       
        lab_count = self.lab_count[layer][j] # 一个字典 lab_count存储每个模板下每种标签有几个样本, 字典格式存储
        if len(lab_count) == 0: # 如果这个字典中一个标签也没有。
            self.lab_count[layer][j][label] = 1 # 将这个标签放入字典中，并计数为1
            self.isclean[layer][j] = label # 将这个模板记为纯净，不纯净的标-1，纯净的标标签编号
        else:
            if label in lab_count: # 如果这个字典中有这个标签
                self.lab_count[layer][j][label] += 1 # 计数加1
            else:
                self.lab_count[layer][j][label] = 1 # 如果这个字典中没有这个标签，且字典还有其他标签。
                self.isclean[layer][j] = -1 # 标为不纯净

    def cmp_and_update(self, xi, layer, label, train_model = 0):
        """
        对比并更新模板

        input:
            xi: 当前样本向量
            layer: 当前层编号 
            label: xi的标签

            train_model: 
                         train_model = 0 为测试当前层 测试时仅需要返回该样本匹配到的模板编号

                         train_model = 1 为训练阶段, 找出相似的模板更新模板, 并返回模板编号

                         train_model = 2  更新j模板的标签状况
        return:
            yes_t: 若匹配成功返回模板编号, 否则返回-1
        """
        no_same = True # 是否不归属于任何模板
        tmps = self.template[layer] # 取出第x层的模板
        Tis = {} # 得分的字典
        for i in range(len(tmps)): # 计算样本和每一个模板的得分
            and_L1 = torch.norm(self.fuzzy_and(xi, tmps[i]), p = 1)
            tmps[i] = tmps[i].float()
            tmp_L1 = torch.norm(tmps[i], p = 1)
            Ti = and_L1/(self.alpha+tmp_L1)
            Tis[i] = Ti # key为模板编号，value为Ti得分

        Tis_sorted = sorted(Tis.items(), key = lambda x: x[1], reverse = True) # 按Ti得分从大到小排序
        
        for j in range(len(Tis_sorted)):
            and_L2 = torch.norm(self.fuzzy_and(xi, tmps[Tis_sorted[j][0]]), p = 1)
            xi = xi.float()
            xi_L2 = torch.norm(xi, p = 1)
            Zj = and_L2/xi_L2

            # print("共振值：" + str(Zj) + "阈值：" + str(self.rou[layer]))
            
            if Zj > self.rou[layer]: 
                # 判定成功，更新模板
                if train_model == 1 and self.isNew[layer][Tis_sorted[j][0]] == 1: 
                    # 只在训练阶段更新模板 train_model = 1 代表训练阶段， 且只能更新
                    print("update template:", Tis_sorted[j][0])
                    self.update_t(xi, layer, Tis_sorted[j][0])

                elif train_model == 2 and self.isCount[layer][Tis_sorted[j][0]] == 1: 
                    # 更新j模板的标签状况
                    print("update label:", Tis_sorted[j][0])
                    self.update_label(xi, layer, Tis_sorted[j][0], label)

                no_same = False
                yes_t = Tis_sorted[j][0] # 返回模板编号
                break

        if no_same: # 创建模板
            yes_t = -1 # 表示没有匹配成功的模板
            if train_model==1:
                yes_t = len(self.template[layer])
                self.create_t(xi, layer, -1, -1) # 无源，无目标的模板

        return yes_t

    def F0(self, input):
        """
        输入矩阵，返回展平，归一，并增加补码编码的张量
        """
        input_ori =input.clone() 
        input = 1-input          
        input_final = torch.cat((input_ori, input), dim =1)
        return input_final

    def forward_m(self, layer): 
        """
        将该层中的每个模板用,其父模板经过网络后的映射结果代替
        """
        # 遍历该层中每个模板
        for j in range(len(self.template[layer])):
            father_idx = self.father_tmp[layer][j] # 取出其父模板
            if self.isclean[layer-1][father_idx] < 0: # 父模板不纯的，都要映射  
                tmp_father = self.template[layer-1][father_idx].detach()
                self.liners[self.linera_layer_num][j] = self.liners[self.linera_layer_num][j].to(self.device)        
                tmp_new = self.liners[self.linera_layer_num][j](tmp_father)
                self.liners[self.linera_layer_num][j] = self.liners[self.linera_layer_num][j].to('cpu')
                self.template[layer][j] = tmp_new #存储映射后的模板
            
    def train_m(self, input, layer=0): 
        """
        通过样本更新模板

        input: 样本矩阵
        layer: 层数
        """
        # 将样本向量映射到0到1之间
        mapping = nn.Sigmoid()
        input = mapping(input)
        # print("shape1: ", input.shape)

        # 使用补码编码扩展向量
        inputs = self.F0(input)
        # print("shape2: ", inputs.shape)
        template_m = [0] * len(inputs)

        for i in range(len(inputs)):
            xi = inputs[i]
            for lay in range(layer + 1):
                if lay == layer: 
                    template_m[i] = self.cmp_and_update(xi, lay, label = -1, train_model = 1) 
                else: 
                    template_m[i] = self.cmp_and_update(xi, lay, label = -1, train_model = 0)
                    if self.isclean[lay][template_m[i]] != -1: # 纯净, 不需要去下层训练了
                        break

        return template_m

    def count_m(self, input, labels, layer=0):
        """
        数据全过一遍，统计模板是否纯净，返回预测的模板编号

        input:样本矩阵
        labels: 对应样本的标签
        layer:层数
        """
        # 将样本向量映射到0到1之间
        mapping = nn.Sigmoid()
        input = mapping(input)

        # 使用补码编码扩展向量
        inputs = self.F0(input)
        yes_ts = []

        for i in range(len(inputs)):
            xi = inputs[i]
            label = labels[i].item()
            for lay in range(layer + 1):
                if lay == layer: 
                    # 统计当前层
                    yes_t = self.cmp_and_update(xi, lay, label, train_model = 2) # 2只统计模板中有哪些标签，不更新模板
                    # print(str(i) + " In the "+ str(lay) + " layer, the template is: "+ str(yes_t))
                    if yes_t == -1:
                         print("In the "+ str(lay) + " layer, no template match: "+str(i))
                else: 
                    # 对于之前的层，传入数据，返回模板编号，根据返回标号的情况进行处理。
                    jl = self.cmp_and_update(xi, lay, label=-1, train_model = 0) # 0 既不更新模板也不统计标签
                    if jl == -1: # 没有所属模板的情况
                        print("In the "+ str(lay) + " layer, no template match: "+str(i))
                        yes_t = -1
                        break
                    if self.isclean[lay][jl] != -1: # 纯净,不需要去下层统计了
                        print("the "+ str(lay) + " tempalte: "+ str(jl) + "pure, self.isclean[" + str(lay) + "][" + str(jl) + "] :" + str(self.isclean[lay][jl]))
                        yes_t = -3 # 表示属于前一层的纯净
                        break
                    print(str(i) + " In the "+ str(lay) + " layer, no template match: "+ str(jl))

            yes_ts.append(yes_t)
        return yes_ts
            
    def test_m(self, input, the_labels, layer=3): # 测试

        # 将样本向量映射到0到1之间
        mapping = nn.Sigmoid()
        input = mapping(input)

        # 使用补码编码扩展向量
        inputs = self.F0(input)
        yes_outs = []
        traget_labels = []

        for i in range(len(inputs)):
            xi = inputs[i]
            yesout = torch.zeros(2*self.ilen)
            traget_label = -1

            for lay in range(layer + 1):
                yes_t = self.cmp_and_update(xi, lay, label=-1, train_model=0)
                if yes_t == -1 and lay != layer: # 不属于任何模板，直接去下一层
                    continue
                else: 
                    if self.isclean[lay][yes_t] == -1 and lay != layer: # 不纯净，去下一层
                        continue
                    else:
                        if yes_t != -1:
                            yesout = self.template[lay][yes_t].float()              
                            labels = self.lab_count[lay][yes_t]
                            labels_sorted = sorted(labels.items(), key = lambda item:item[1], reverse = True)
                            # 找出对应模板的标签
                            print("The template matches the sample is: in ", str(lay), " layer, the ", str(yes_t)," template")
                            for i in range(len(labels_sorted)):
                                traget_label = labels_sorted[i][0]
                                break
                    
                            if traget_label != -1:
                                yesout = self.label_templates[traget_label]
                            break
                        print("The template matches the sample is: in ", str(lay), " layer, the ", str(yes_t)," template")
                        
            yes_outs.append(yesout)
            traget_labels.append(traget_label)

        # print("traget_labels: ", traget_labels)
        return yes_outs, traget_labels
        
    def init_next_tmp(self, layerlast=0):
        """
        复制(其实不是复制, 是创建了一个全新的模板, 跟之前的一样, 顺便在这个时候设置了主要标签)模板到下一层，并加入一个初始映射层(即自动添加线性层)
        layerlast: 上一层的编号
        """
        count_new = 0 # 模板计数。用于记录下一层中一共有多少个模板。
        for jlast in range(len(self.template[layerlast])): # 取出上一层的所有模板

            tmp = self.template[layerlast][jlast]
            if self.isclean[layerlast][jlast] != -1 | self.isclean[layerlast][jlast] != -2: # 模板纯净,不需要复制到下一层
                continue
            else : # 不纯模板
                labels = self.lab_count[layerlast][jlast]
                for key in labels.keys():
                    print("In layer ", layerlast, " template: ", jlast, " key: ", key, " number is : ", labels[key])
                    if labels[key] >= 10:
                        print("number bigger than 10")
                        traget_label = key
                        print("father template: (", layerlast,"," ,jlast, ")")
                        self.create_t(tmp, layerlast + 1, jlast, traget_label) # 创建新一层的模板
                        self.liners[self.linera_layer_num][len(self.liners[self.linera_layer_num])] = T_change(2*self.ilen, 2*self.ilen)
                        count_new +=1 
                    else:
                        print("number smaller than 10")
                        
        print(str(count_new) + " templates have been copied to " + str(layerlast + 1) + "layer")           

    def parameters(self, layer=0):
        for j in range(len(self.liners[layer])):
            for i in range(len(self.liners[layer][j].parameters())):
                yield self.liners[layer][j].parameters()[i]
                
    def change_isNew(self, layer):
        for i in range(len(self.isNew[layer])):
            self.isNew[layer][i] = 0

    def change_isCount(self, layer):
        for i in range(len(self.isCount[layer])):
            self.isCount[layer][i] = 0

    def get_average(self):

        label_templates_statistic = []

        for i in range(self.class_num):
            label_templates_statistic.append([])
            self.label_templates.append(0)

        for lay in range(self.layer_num):
            for j in range(len(self.template[lay])):
                if self.isclean[lay][j] != -1 or lay == self.layer_num-1:
                    labels = self.lab_count[lay][j]
                    labels_sorted = sorted(labels.items(), key = lambda item:item[1], reverse = True)
                    for i in range(len(labels_sorted)):
                        traget_label = labels_sorted[i][0]
                        break
                    label_templates_statistic[traget_label].append(self.template[lay][j])
        
        for i in range(len(label_templates_statistic)):
            sum_of_the_class = torch.zeros(2*self.ilen)
            sum_of_the_class = sum_of_the_class.to(self.device)
            count = 0
            for j in range(len(label_templates_statistic[i])):
                sum_of_the_class += label_templates_statistic[i][j].detach()
                count += 1
            avg_template = sum_of_the_class/count
            self.label_templates[i] = avg_template

    

