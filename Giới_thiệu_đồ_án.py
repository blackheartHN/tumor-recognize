"""
Giới thiệu về đồ án "Ứng dụng Máy học phát hiện u não".

Cung cấp thông tin tổng quan về đồ án, bao gồm:
- Đặt vấn đề và giải pháp
- Lựa chọn mô hình
- Giới thiệu dataset
- Quá trình training
- Các thông số của mô hình

Cung cấp liên kết đến các trang khác như:
- Trang "Mô hình máy học" để hiển thị mô hình và cho phép người dùng thử nghiệm
- Trang "Tham khảo" để liệt kê các tài liệu tham khảo

Cách sử dụng:
1. streamlit run Giới_thiệu_đồ_án.py
2. Xem thông tin giới thiệu về đồ án
3. Truy cập các trang khác thông qua sidebar
"""
import streamlit as st
import os


current_dir = os.path.dirname(__file__)

# Construct the relative path to the model file
images_path = os.path.join(current_dir, '../images/')

def introduction():
    st.header("Giới thiệu đồ án")
    st.subheader("Đặt vấn đề")
    st.write("""
    - U não được coi là một trong những căn bệnh nguy hiểm ở cả trẻ em và người lớn. 
    - Mỗi năm, có đến hàng trăm nghìn người được chẩn đoán mắc căn bệnh nguy hiểm này, và tỉ lệ tử vong vô cùng cao.
    - Để chẩn đoán được 1 khối u não, ta cần phải phân tích dựa trên ảnh MRI của các bệnh nhân. Tuy nhiên, chính vì việc này có độ phức tạp cao, nên cần phải có 1 bác sĩ lành nghề để phân tích ảnh MRI của các bệnh nhân. 
    - Việc thiếu các bác sĩ có kinh nghiệm sẽ khiến việc lập báo cáo MRI và phân tích trở nên khó khăn và tốn thời gian, và việc đào tạo các bác sĩ mới cũng không kém phần khó khăn. 
    - Và đôi lúc các bác sĩ cũng có thể mắc sai sót do mức độ phức tạp liên quan đến khối u não và đặc tính của chúng.
            """)
    st.subheader("Giải pháp")
    st.write("""
    - Việc xây dựng 1 ứng dụng máy học phát hiện u não 1 cách tự động sẽ giúp ích rất nhiều trong vấn đề này
    - Ứng dụng này sẽ được train trên một bộ dữ liệu lớn gồm các hình ảnh MRI của não từ các bệnh nhân đã được chẩn đoán có u não và không có u não, và từ đó tự động phân tích hình ảnh từ MRI và đưa ra các dự đoán, giúp tiết kiệm thời gian và hỗ trợ các bác sĩ trong quá trình chẩn đoán bệnh.       
    - Ngoài ra, ứng dụng này cũng sẽ hỗ trợ các sinh viên trường Y trong việc học tập và rèn luyện, tích lũy kinh nghiệm, giúp họ hiểu rõ hơn về các dạng khối u não và phương pháp chẩn đoán.
             """)
    st.subheader("Lựa chọn mô hình")
    st.write("""
    - Ứng dụng được xây dựng tại đồ án này sử dụng mô hình **Convolutional Neural Network (CNN)**.
    - CNNs được sử dụng rộng rãi trong xử lý hình ảnh với khả năng tốt trong việc trích xuất và học các đặc trưng từ dữ liệu hình ảnh. Chúng có khả năng học cấu trúc của hình ảnh 1 cách tự động và có thể cải thiện hiệu suất trong việc nhận diện các đối tượng phức tạp như khối u não.
    - Ngoài ra, thư viện **TensorFlow** có cung cấp các lớp **Convolution** và **MaxPooling** giúp giảm kích thước không gian của hình ảnh, giữ lại thông tin quan trọng và loại bỏ thông tin không cần thiết, giúp giảm thời gian train và tiết kiệm tài nguyên máy tính.
    - CNNs còn có thể được train lại với các tập dữ liệu mới, giúp cải thiện độ chính xác và hiệu quả chẩn đoán theo thời gian.
             """)
    st.subheader("Giới thiệu dataset")
    st.write("""
    - Bộ dataset dùng để train model trong ứng dụng máy học này là [**Br35H :: Brain Tumor Detection 2020**](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection?select=no)
    - Br35H :: Brain Tumor Detection 2020 bao gồm 3000 ảnh, trong đó có 1500 ảnh xuất hiện triệu chứng u não, và 1500 ảnh bình thường.    
             """)
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\dataset_preview_yes.PNG", caption="Folder Yes chứa 1500 ảnh xuất hiện triệu chứng u não.")
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\dataset_preview_no.PNG", caption="Folder No chứa 1500 ảnh bình thường.")
    st.write("""
    - Để tiến hành train cho model, bộ dataset được chia như sau:
        - Tập train: gồm 1050 ảnh bình thường và 1050 ảnh u não.
        - Tập val: gồm 400 ảnh bình thường và 400 ảnh u não.
        - Tập test: gồm 50 ảnh bình thường và 50 ảnh u não.
             """)

    st.subheader("Quá trình training")
    st.write("""
    - **Tiền xử lý dữ liệu**:
        - **Chuẩn hóa**: Lớp Rescaling được sử dụng để chuẩn hóa các giá trị pixel trong tập train từ khoảng [0, 255] xuống [0, 1]. 
        - **Caching**: Lưu trữ các mẫu dữ liệu trong bộ nhớ cache, giúp tăng tốc độ của quá trình huấn luyện.
        - **Prefetching**: Tải dữ liệu cho các batch tiếp theo trong quá trình huấn luyện, giúp giảm thiểu thời gian chờ đợi.
    - **Xây dựng model**:
        - Sử dụng model **Sequential**, là một chuỗi các lớp được xếp theo hàng ngang, 
        - Tiếp theo là các lớp **Conv2D** và **MaxPooling2D**, giúp trích xuất và tổng hợp đặc trưng từ hình ảnh.
        - Sau đó, các lớp này được lặp lại để tăng cường khả năng học cấu trúc của dữ liệu hình ảnh.
        - Cuối cùng, một loạt các lớp **Dense** được sử dụng để biến đổi các đặc trưng thành dự đoán xác suất cho các lớp.     
             """)
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\model_architecture.PNG", caption="Cấu trúc của mô hình.")
    st.write("""
    - **Huấn luyện model**:
        - Sử dụng optimizer **Adam** (optimizer='adam') để huấn luyện model. Đây là một thuật toán hiệu quả thích ứng tốc độ học tập cho từng tham số trong quá trình huấn luyện.  
        - Hàm mất mát **SparseCategoricalCrossentropy** phù hợp cho các bài toán phân loại đa lớp. Nó đo lường sự khác biệt giữa xác suất lớp dự đoán và nhãn thực tế.  
        - Quá trình huấn luyện diễn ra trong 8 epochs.  
             """)
    st.subheader("Các thông số của mô hình")
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\loss_epoch.PNG", caption="Giá trị hàm mất mát theo từng epoch.")  
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\accuracy_epoch.PNG", caption="Accuracy theo từng epoch.")  
    st.image("D:\\proj\\IE221.PYTHON\\final\\images\\consfusion_matrix.PNG", caption="Ma trận nhầm lẫn.")  


introduction()


def authenticated_menu():
    st.sidebar.page_link("pages/Mô_hình_máy_học.py")
    st.sidebar.page_link("pages/Reference.py")