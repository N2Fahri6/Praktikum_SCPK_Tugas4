% 123190024 - Naufal Nur Fahriza
% Jaringan Saraf Tiruan - Perceptron dengan pola fungsi logika "or" menggunakan 2 variabel
% Dengan diketahui bobot awal [-1,1] dan bias [1]

disp("Pola input-an: ")
p1 = [1;1];
p2 = [1;0];
p3 = [0;1];
p4 = [0;0];
P = [p1 p2 p3 p4]

disp("Target : ")
t1 = 1;
t2 = 1;
t3 = 1;
t4 = 0;
T = [t1 t2 t3 t4]

% Menentukan Target dengan perceptron baru
net = newp([0 1;0 1],1);

%Bobot Awal
Bobot = [-1 1]
net.IW{1,1} = Bobot;

%Bias Awal
Bias = [1]
net.b{1} = Bias;

disp("Tampilan Output : ")

disp("Proses pelatihan input-an 1 Variabel1 = 1 & Variabel2 = 1 ")
Output1 = sim(net,p1)
Error1 = t1 - Output1
disp("Variabel dW (Menyimpan Perubahan Bobot)")
dW = learnp(Bobot,p1,[],[],[],[],Error1,[],[],[],[],[])
% Bobot Akhir
disp("Nilai Bobot akhir dilakukan penjumlahan -> Bobot + dW ")
Bobot = Bobot + dW

disp("Proses pelatihan input-an 2 Variabel1 = 1 & Variabel2 = 0")
Output2 = sim(net,p2)
Error2 = t2 - Output2
dW = learnp(Bobot,p2,[],[],[],[],Error2,[],[],[],[],[])
Bobot = Bobot + dW

disp("Proses pelatihan input-an 3 Variabel1 = 0 & Variabel2 = 1")
Output3 = sim(net,p3)
Error3 = t3 - Output3
dW = learnp(Bobot,p3,[],[],[],[],Error3,[],[],[],[],[])
Bobot = Bobot + dW

disp("Proses pelatihan input-an 4 Variabel1 = 0 & Variabel2 = 0")
Output4 = sim(net,p4)
Error4 = t4 - Output4
dW = learnp(Bobot,p4,[],[],[],[],Error4,[],[],[],[],[])
Bobot = Bobot + dW

disp("Proses pelatihan input-an semuanya")
Output = sim(net,P)
Target = T
Error = T-Output
Performance = perform(net,T,Output,{1})

% Proses pelatihan ulang dengan fungsi "net = train(net,P,T)"
net = train(net,P,T);
    
disp("Hasil  akhir proses pelatihan input-an dalam keseluruhan : ")
Output = sim (net, P)
Target = T
Error = T-Output
Performance = perform(net,T,Output,{1})

% Menampilkan Bobot dan Bias dalam keadaan optimal Pada kedua variabel
% tersebut
disp("Bobot Dalam Bentuk Optimal : ")
disp(net.IW{1,1})
disp("Bias Dalam Bentuk Optimal : ")
disp(net.b{1})

disp("Menampilkan GUI proses pelatihan & performance !!!")
