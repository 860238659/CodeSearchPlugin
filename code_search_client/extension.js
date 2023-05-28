const vscode = require('vscode');
const axios = require('axios');


/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
	console.log('Congratulations, your extension "code_search" is now active!');

	async function search(query) {
		const url = `http://127.0.0.1:5000?searchKey=${query}`;

		let items;

		try {
			const response = await axios.post(url);
			const list = response.data;
			console.log("成功获取数据")
			items = list.map((item, index) => {
				const code = item;
				return {
					label: `候选code ${index + 1}`,
					description: code
				}
			});
			console.log(items)
			return items;
		} catch (error) {
			console.error(error);
			vscode.window.showErrorMessage('An error occurred while sending the selected text.');
		}
	}

	function displayResults(query) {
		// 创建一个 QuickPick 对象
		const quickPick = vscode.window.createQuickPick();
		quickPick.placeholder = '请选择需要复制的文本项';

		// 设置 QuickPick 的候选项数组
		quickPick.items = query;

		// 当用户选择一项时触发的回调函数
		quickPick.onDidChangeSelection(([selection]) => {
			if (selection) {
				// 获取当前选中项的文本内容
				const text = selection.description;

				// 创建一个命令对象
				const command = {
					title: '复制文本到剪贴板',
					command: 'extension.copyTextToClipboard',
					tooltip: 'Click to copy the text to clipboard',
					arguments: [text],
				};

				// 显示复制命令按钮
				vscode.window.showInformationMessage('已选中 \n' + text, command).then((selected) => {
					if (selected) {
						// 如果用户点击了复制命令按钮，则将文本内容写入剪贴板
						vscode.env.clipboard.writeText(text);
						vscode.window.showInformationMessage(`已经成功复制了 "${text}"`);
					}
				});
			}
		});

		quickPick.show();
	}

	let disposable = vscode.commands.registerCommand('try.getSelectedText', async function () {
			// 可以使用 vscode.window.activeTextEditor.document.getText(selection) 获取当前选中区域的文本。
			const editor = vscode.window.activeTextEditor;
			if (!editor) {
				vscode.window.showErrorMessage('没有获得输入')
				return
			}

			const selection = editor.selection;
			if (selection.isEmpty) {
				vscode.window.showInformationMessage('没有文本输入')
			}

			const selectedText = editor.document.getText(selection);
			console.log(selectedText)
			// Display a message box to the user
			vscode.window.showInformationMessage("正在查找相关代码，请稍候。");

			// 获取 memento 对象
			const memento = context.workspaceState;

			// 在缓存中查找特定的数据
			let result = memento.get(selectedText);
			if (result) {
				console.log(result)
				console.log("从缓存中找到了")
				displayResults(result)
			} else {
				result = await search(selectedText)
				// 将数据添加到缓存中
				memento.update(selectedText, result);
				console.log("添加到缓存中啦")
				displayResults(result)
			}
		}
	);

}

// This method is called when your extension is deactivated
function deactivate() {
	console.log('"code_search"已经停止运行！')
}

module.exports = {
	activate,
	deactivate
}
