import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
// import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1


Item {
    id : root
    Layout.fillHeight : true
    Layout.fillWidth : true

    property var max_file_count: 99;

    Component {
        id : delegate_component

        Item {
            height : 50
            width : list_view.width - 25

            property alias path : list_view_text.text
            property alias type : list_view_combobox.currentText

            RowLayout {
                width : parent.width
                spacing : 10

                Button {
                    id : list_view_delete_button
                    implicitWidth : 40

                    icon.source : '../../resource/delete_outline_black_24dp.svg'
                    icon.height : 20

                    onClicked : {
                        list_view.model.remove(index);
                        con.delete_file(list_view_text.text);
                    }
                }

                TextField {
                    id : list_view_text
                    objectName : 'path'

                    Layout.fillWidth : true
                    Layout.fillHeight : true
                    text : list_text
                    // readOnly : true
                }

                ComboBox {
                    id : list_view_combobox
                    objectName : 'type'
                    model : [
                        '-',
                        'Damping',
                        'Stiffness',
                        'Internal Load',
                        'External Load',
                        'Target Nodes'
                    ]

                    enabled : rb_matrix.checked
                    implicitWidth : 180
                    onCurrentIndexChanged : {
                        con.select_file_and_type(list_view_text.text + '|' + currentIndex);
                    }
                }

                Rectangle {
                    width : 20
                    visible : false
                }
            }
        }
    }

    FileDialog {
        id : file_dialog

        nameFilters : ['All files (*.*)']
        folder : StandardPaths.standardLocations(StandardPaths.DocumentsLocation)[0]

        // // for Qt.labs.platform
        fileMode : FileDialog.OpenFiles

        // // for QtQuick.Dialogs
        // folder : shortcuts.documents
        // selectMultiple : true

        onAccepted : {
            let already_selected = []; // 이미 선택된 파일 목록
            for (var idxModel = 0; idxModel < list_model.rowCount(); ++ idxModel) {
                already_selected.push(list_model.get(idxModel).list_text);
            }

            for (var idxUrl = 0; idxUrl < files.length; ++ idxUrl) {
                if (max_file_count <= list_view.count) {
                    break; // 최대 선택 파일 개수 제한
                }

                var path = files[idxUrl].replace('file:///', '')

                if (! already_selected.includes(path)) {
                    list_model.append({"list_text": path});
                } else { // 이미 선택된 파일 넘김
                    con.log('DEBUG|`' + path + '` already selected');
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill : parent
        spacing : 0

        RowLayout {
            spacing : 10

            RadioButton {
                id : rb_matrix
                text : 'Read Matrix'
                checked : true

                ToolTip.visible : hovered
                ToolTip.text : '모델을 구성할 행렬 파일을 읽습니다.'
            }

            RadioButton {
                text : 'Read Model'

                ToolTip.visible : hovered
                ToolTip.text : '기존에 구성한 모델을 불러옵니다.'

                onCheckedChanged : {
                    list_model.clear();
                    con.delete_all_files();

                    if (checked) { // 모델 파일 선택
                        max_file_count = 1;
                        file_dialog.fileMode = FileDialog.OpenFile;
                    } else { // matrix 파일 선택
                        max_file_count = 99;
                        file_dialog.fileMode = FileDialog.OpenFiles;
                    }
                }
            }

            Button {
                text : 'Add File'
                icon.source : '../../resource/add_black_24dp.svg'
                highlighted : true
                font.capitalization : Font.Capitalize

                onReleased : {
                    file_dialog.open()
                }
            }

            Button {
                text : 'Load'
                // TODO icon
                highlighted : list_model.count
                font.capitalization : Font.Capitalize

                onReleased : {
                    if (rb_matrix.checked) {
                        con.read_matrices();
                    } else {
                        con.read_model_from_selected();
                    }
                }
            }
        }

        Rectangle {
            id : rectangle
            Layout.fillHeight : true
            Layout.fillWidth : true
            color : "#F2F2F2"

            ScrollView {
                anchors.fill : parent
                anchors.margins : 10
                clip : true

                // ScrollBar.vertical.policy : ScrollBar.AlwaysOn
                ScrollBar.vertical.policy : ScrollBar.AsNeeded
                ScrollBar.horizontal.policy : ScrollBar.AlwaysOff

                ListView {
                    id : list_view
                    objectName : 'file_list_view'

                    Layout.fillWidth : true

                    model : ListModel {
                        id : list_model
                    }
                    delegate : delegate_component
                }
            }
        }
    }
}
