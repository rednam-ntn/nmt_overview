### __Cơ chế Tối ưu hóa__

* Thuật toán tối ưu hóa (optimizer): ta có 2 sự lựa chọn là Adam ([Kingma & Ba, 2015](https://arxiv.org/abs/1412.6980)) và SGD (stochastic gradient descent). Mặc dù Adam mới được ra mắt gần đây và được cộng đồng nghiên cứu NLP sử dụng thường xuyên vì sự vượt trội rõ ràng của Adam so với SGD thì một số nghiên cứu đã cho thấy là với nhưng tùy chỉnh hợp lý thì SGD đã có thể nhỉnh  hơn Adam một chút ([Wu et al., 2016](https://arxiv.org/abs/1609.08144)) hoặc vượt trội hơn hẳn với việc áp dụng momentum ([Zhang & Mitliagkas, 2017](https://arxiv.org/abs/1706.03471)).
* Cơ chế khởi động tốc độ học ban đầu và giảm dần khi về cuối (warmup_scheme & decay_scheme). Mô hình cho phép lựa chọn một số cơ chế tích hợp sẵn như là:
    * Tensor2Tensor's warmup_scheme, khởi đầu với tốc độ học lr nhỏ hơn 100 lần và tăng dần cho đến khi đạt được con số mong muốn.
    * luong234 decay_scheme, sau 2/3 bước huấn luyện, bắt đầu giảm tốc độ học lr 4 lần, mỗi lần 50%
    * luong5 decay_scheme, sau 1/2 bước huấn luyện, bắt đầu giảm tốc độ học lr 5 lần, mỗi lần 50%
    * luong10 decay_scheme, sau 1/2 bước huấn luyện, bắt đầu giảm tốc độ học lr 10 lần, mỗi lần 50%
* Bước huấn luyện (trainnig_step): Dao động phụ thuộc vào độ phức tạp và khả năng huấn luyện, thường từ 10.000 lần cho tới hàng trăm nghìn lần.