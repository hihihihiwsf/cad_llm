import matplotlib.pyplot as plt

# Data for plotting
input_token_ratio = [20, 40, 60, 80]  # X-axis data
cad_any=[43.9,65.7,73.0,73.8]  # Y-axis data
virt_any=[34.1,40.8,41.4,42.4] 

cad_full=[1.19, 10.8, 31.1, 49.5]
virt_full=[0.059, 0.71,3.5,14.6]

cad_f1=[16.9, 36.4, 53.2, 62.7]
virt_f1=[8.5,11.0,11.9,12.5]

# Plotting the data
plt.plot(input_token_ratio, virt_f1, marker='o', label='Vitruvion')
plt.plot(input_token_ratio, cad_f1, marker='*', label='CAD-LIP')

# # Adding labels for each y-value
# for i in range(len(hit_rate)):
#     plt.text(input_token_ratio[i], hit_rate[i] + 1, f"{hit_rate[i]:.2f}%", ha='center')

# Adding labels and title
plt.xlabel('Input entity ratio (%)',fontsize=14)
plt.ylabel('CAD F1 (%)',fontsize=14)

# Setting x-axis ticks
plt.xticks([20, 40, 60, 80])
plt.ylim(0, 100)

# Adding a legend
plt.legend(fontsize=14)

# Display grid
plt.grid(False)

# Save the figure as a PDF
plt.savefig('ratio_f1_acc.pdf', bbox_inches='tight')

