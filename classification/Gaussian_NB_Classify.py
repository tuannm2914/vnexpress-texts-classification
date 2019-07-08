import numpy as np
from create_dicts import *
import os
import re

number_document = 33579

class Statistic:
    def __init__(self,name,class_file,common_bow):
        self.class_name = name
        self.class_file = class_file
        self.common_bow = common_bow

    def vectorize_data(self,line):
        data = line.strip().split(" ")
        dict_words = create_text_dict(data)
        return [dict_words[key] if key in dict_words else 0 for key,value in self.common_bow.items()]

    def vectorize_class(self):
        class_data = []
        with open(self.class_file) as fp:
            for line in fp:
                class_data.append(self.vectorize_data(line))
        return np.array(class_data)

    def mean_statistic(self):
        class_data = self.vectorize_class()
        class_data = np.array(class_data)
        mean_data = np.divide(np.sum(class_data,axis = 0 ) , class_data.shape[0])
        return mean_data

    def dev_statistic(self):
        mean_data = self.mean_statistic()
        class_data = self.vectorize_class()
        dev_data = np.sqrt(np.divide(np.sum(np.power((np.subtract(class_data,mean_data)) ,2),axis = 0),len(class_data)))
        return dev_data



class GNB:
    def __init__(self,data,*Statistic):
        self.list_statistic = Statistic
        self.data = data

    def vectorize_data(self):
        return self.list_statistic[0].vectorize_data(self.data)

    def get_pdf(self,data,std,mean):
        return  1 / (np.sqrt(np.multiply(2*3.14 ,np.power(std,2))) * np.exp(-np.divide((np.power(np.subtract(vectorized_data ,mean), 2) , np.multiply(2 ,np.power(mean,2))))))



if __name__ == "__main__":
    topic_path = "/home/tuannm/mine/vnexpress-texts-classification/data/tokenized_data"
    common_bow = get_entire_bow(topic_path)
    m = []
    d = []
    n = []
    vectorized_data = []
    j = 0
    for pathfile in os.listdir(topic_path):
        if j < 2:
            mean = []
            std = []
            class_file = os.path.join(topic_path,pathfile)
            class_name = re.sub('.txt',"",pathfile)
            cls1 = Statistic(class_name,class_file,common_bow)
            full_mean = cls1.mean_statistic()
            full_std = cls1.dev_statistic()
            s = "Cơ_quan tự đảm_bảo lương biên_chế vượt định_mức Bộ Nội_vụ , Tài_chính và Lao_động Thương_binh và Xã_hội vừa ban_hành 11 thông_tư hướng_dẫn thực_hiện chế_độ lương mới đối_với công_chức , viên_chức và lực_lượng vũ_trang . Theo thông_tư 02 của Bộ Tài_chính , đối_với biên_chế cán_bộ chuyên_trách , công_chức cấp xã tăng thêm so với thời_điểm báo_cáo nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương , nếu trong phạm_vi tổng mức biên_chế được giao hoặc phê_duyệt thì nhu_cầu này được tính vào kinh_phí cải_cách tiền_lương của kỳ sau . Nếu số biên_chế vượt tổng mức thì nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương của số này do cơ_quan tự đảm_bảo , không được tổng_hợp vào nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương của các Bộ , cơ_quan ngang bộ và các tỉnh , thành_phố . Nguồn để chi_trả tiền_lương tăng thêm cho số lao_động không thuộc chỉ_tiêu biên_chế nằm trong phạm_vi dự_toán ngân_sách hàng năm và nguồn thu để lại theo chế_độ cho cơ_quan . Số tăng thêm này không được tổng_hợp vào nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương của các Bộ , cơ_quan trung_ương và tỉnh , thành_phố . Riêng tiền_lương tăng thêm đối_với lao_động của các đơn_vị sự_nghiệp có thu , tự đảm_bảo toàn_bộ kinh_phí hoạt_động thường_xuyên thì đơn_vị được để lại nguồn thu theo chế_độ để chi_trả . Nếu nguồn thu không đủ chi_trả lương tăng thêm thì phần thiếu được ngân_sách đảm_bảo và nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương được tính vào kinh_phí thực_hiện cải_cách tiền_lương của Bộ , cơ_quan trung_ương và tỉnh , thành_phố . Thông_tư ghi rõ , tiền_lương tăng thêm đối_với biên_chế , lao_động của hệ_thống công_đoàn các cấp thì được chi_trả từ nguồn 2 % kinh_phí công_đoàn ; không tổng_hợp vào nhu_cầu kinh_phí thực_hiện cải_cách tiền_lương của các Bộ , cơ_quan trung_ương và tỉnh , thành_phố . Cũng tại thông_tư này , Bộ Tài_chính nêu rõ 3 nguồn kinh_phí thực_hiện cải_cách tiền_lương gồm : tối_thiểu 40 % số thu được để lại theo chế_độ quy_định của các đơn_vị sự_nghiệp có thu và các cơ_quan hành_chính có thu ( riêng đơn_vị thuộc ngành y_tế , tỷ_lệ này là 35 % ) ; 10 % tiết_kiệm chi thường_xuyên ; 50 % số tăng thu_ngân sách địa_phương . Công_chức được tăng lương trước thời_hạn 12 tháng nếu lập thành_tích xuất_sắc . Đó là điểm mới trong thông_tư 03 của Bộ Nội_vụ về hướng_dẫn thực_hiện chế_độ nâng bậc lương thường_xuyên và nâng bậc lương trước thời_hạn . Theo đó , nếu đã được thẩm_quyền quyết_định công_nhận bằng văn_bản , chưa xếp vào bậc lương cuối_cùng trong ngạch hoặc chức_danh và còn thiếu 1-12 tháng thì cá_nhân đó được xét nâng một bậc lương trước thời_hạn . Tỷ_lệ được nâng bậc lương trước thời_hạn do lập thành_tích xuất_sắc trong 1 năm không quá 5 % tổng biên_chế trả lương của cơ_quan . Việc_làm này không được quá 2 lần trong thời_gian giữ một bậc lương . Đối_với cán_bộ công_chức , viên_chức chưa được xếp bậc lương cuối_cùng trong ngạch hoặc trong chức_danh và có thông_báo nghỉ hưu thì chế_độ nâng lương được tính như sau : Nếu trong thời_gian giữ bậc hoàn_thành nhiệm_vụ được giao , không bị kỷ_luật và tại thời_điểm có thông_báo nghỉ hưu còn thiếu 1-12 tháng để được nâng lương thì được nâng một bậc trước thời_hạn quy_định . Thông_tư 03 cũng quy_định , đối_với chuyên_gia cao_cấp , nếu chưa xếp bậc lương cuối_cùng trong bảng thì sau đủ 60 tháng được xét nâng một bậc . Cán_bộ công_chức , viên_chức nếu chưa xếp bậc lương cuối_cùng trong ngạch công_chức , viên_chức từ_loại A0 đến A2 thì sau đủ 36 tháng được xét nâng một bậc lương . Cán_bộ công_chức , viên_chức nếu chưa xếp bậc lương cuối_cùng trong ngạch công_chức , viên_chức loại B và C thì sau đủ 24 tháng được xét nâng lên một bậc . Phụ_cấp trách_nhiệm công_việc tối_đa 145.000 đồng ; phụ_cấp lưu_động tối_đa 174.000 đồng . Thông_tư 05 của Bộ Nội_vụ hướng_dẫn thực_hiện chế_độ phụ_cấp trách_nhiệm công_việc , có 4 mức phụ_cấp , thấp nhất_là 29.000 đồng . Còn theo thông_tư 06 hướng_dẫn thực_hiện chế_độ phụ_cấp lưu_động , có 3 mức phụ_cấp , tối_thiểu là 58.000 đồng . Làm thêm giờ vào ngày nghỉ lễ được hưởng gấp 3 lương so với ngày thường . Theo thông_tư 08 của liên_bộ Nội_vụ và Tài_chính , tiền_lương làm_việc vào ban_đêm sẽ bằng tiền_lương giờ làm_việc theo tiêu_chuẩn nhân với 130 % và nhân với số giờ làm_việc ban_đêm . Nếu làm thêm giờ vào ban_ngày , tiền_lương làm thêm sẽ bằng tiền_lương giờ làm_việc theo tiêu_chuẩn , nhân với 150 % ( hoặc 200 - 300 % ) , nhân với số giờ thực_tế làm thêm . Trong đó , mức 150 % áp_dụng với ngày thường , 200 % với ngày nghỉ hàng tuần và 300 % với ngày lễ hoặc nghỉ bù ngày lễ . Trước đó , ngày 5/1 , Bộ Lao_động Thương_binh và Xã_hội cũng ban_hành 11 thông_tư hướng_dẫn chế_độ lương mới đối_với các chức_danh trong công_ty nhà_nước"
            data = cls1.vectorize_data(s)
            for i in range(len(data)):
                if data[i] != 0:
                    mean.append(full_mean[i])
                    std.append(full_std[i])
                    if j == 0:
                        vectorized_data.append(data[i])
            print(mean)
            m.append(mean)
            d.append(std)
            n.append(class_name)
            j+= 1
    mean_mean = np.mean(np.mean(m))
    print(mean_mean)
    for i in range(len(n)):
        print(n[i])
        mean = m[i]

        mean = np.array(mean)
        std = d[i]
        std  = np.array(std)

        x = np.array(vectorized_data)
        print(np.sum(np.log(np.multiply(np.divide(mean_mean, (np.sqrt(np.multiply(2*3.14,np.power(std,2)))) ), np.exp(-np.divide((np.power(np.subtract(x,mean),2)),np.multiply(2,np.power(std,2))))))))