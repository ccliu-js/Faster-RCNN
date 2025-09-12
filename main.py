# 创建模型实例
model = MyFasterRCNN(num_classes=6)

# 训练
model.train()
loss_dict = model(images, targets)
loss = sum(loss for loss in loss_dict.values())
loss.backward()
optimizer.step()

# 保存最优模型
model.save("weight/best_model.pth")

# 推理
model.load("weight/best_model.pth")
model.eval()
outputs = model(images)  # 直接推理


"""
1.明天上午完成对模型的封装  -- 习惯和理解这种封装




2.能够随时调整模型的内部特征  --以及做好内部特征的可视化    




3.会使用保存的参数进行新的训练----相当于在原来训练的基础上，进行进一步的训练


"""