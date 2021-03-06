import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1


Item {
    id : root
    Layout.fillHeight : true
    Layout.fillWidth : true

    property var max_file_count: 99;

    ButtonGroup {
        id : file_group
    }

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

                // Delete button
                Button {
                    id : list_view_delete_button
                    implicitWidth : 40
                    enabled : !rb_reference.checked

                    RowLayout {
                        anchors.fill : parent
                        Text {
                            text : '\ue14c'
                            font.family : 'Material Icons Outlined'
                            font.pointSize : 20
                            Layout.alignment : Qt.AlignCenter
                        }
                    }

                    onClicked : {
                        list_model.remove(index);
                        con.delete_file(list_view_text.text);
                    }
                }

                // Path text field
                TextField {
                    id : list_view_text
                    objectName : 'path'

                    Layout.fillWidth : true
                    Layout.fillHeight : true
                    text : list_text
                    readOnly : rb_reference.checked
                }

                // File type combo box
                ComboBox {
                    id : list_view_combobox
                    objectName : 'type'
                    model : [
                        '-',
                        'Capacitance Matrix',
                        'Conductance Matrix',
                        'Internal Solicitation Matrix',
                        'External Solicitation Matrix',
                        'Target Nodes'
                    ]

                    enabled : rb_matrix.checked
                    implicitWidth : 250
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

        nameFilters : ['All files (*)']
        // folder : StandardPaths.standardLocations(StandardPaths.DocumentsLocation)[0]

        fileMode : FileDialog.OpenFiles // Qt.labs.platform

        onAccepted : {
            let already_selected = []; // ?????? ????????? ?????? ??????
            for (var idxModel = 0; idxModel < list_model.rowCount(); ++ idxModel) {
                already_selected.push(list_model.get(idxModel).list_text);
            }

            for (var idxUrl = 0; idxUrl < files.length; ++ idxUrl) {
                if (max_file_count <= list_view.count) {
                    break; // ?????? ?????? ?????? ?????? ??????
                }

                var path = files[idxUrl].replace('file:///', '')

                if (! already_selected.includes(path)) {
                    list_model.append({"list_text": path});
                } else { // ?????? ????????? ?????? ??????
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
                id : rb_reference
                ButtonGroup.group : file_group
                text : 'Reference Models'

                ToolTip.visible : hovered
                ToolTip.text : '????????? ?????? ????????? ???????????????.'

                onCheckedChanged : {
                    if (checked) {
                        list_model.clear();
                        con.update_reference_models();
                    }
                }
            }

            RadioButton {
                id : rb_matrix
                ButtonGroup.group : file_group
                text : 'Read Matrix'

                ToolTip.visible : hovered
                ToolTip.text : '????????? ????????? ?????? ????????? ????????????.'

                onCheckedChanged : {
                    app.from_matrices(checked);

                    if (checked) {
                        list_model.clear();
                        con.delete_all_files();

                        max_file_count = 99;
                        file_dialog.fileMode = FileDialog.OpenFiles;
                    }
                }
            }

            RadioButton {
                id : rb_model
                ButtonGroup.group : file_group
                text : 'Read Model'

                ToolTip.visible : hovered
                ToolTip.text : '????????? ????????? ????????? ???????????????.'

                onCheckedChanged : {
                    if (checked) {
                        list_model.clear();
                        con.delete_all_files();

                        max_file_count = 1;
                        file_dialog.fileMode = FileDialog.OpenFile;
                    }
                }
            }

            Button {
                id : _af
                highlighted : rb_matrix.checked || rb_model.checked
                font.capitalization : Font.Capitalize
                implicitWidth : _af_layout.implicitWidth + 10

                ToolTip.visible : hovered
                ToolTip.text : '?????? ?????? ?????? ??????'

                RowLayout {
                    id : _af_layout
                    anchors.fill : parent

                    Text {
                        text : '\ue145'
                        font.family : 'Material Icons Outlined'
                        font.pointSize : _af.font.pointSize + 4
                        Layout.alignment : Qt.AlignRight
                    }

                    Text {
                        text : 'Add File'
                        font : _af.font
                        Layout.alignment : Qt.AlignLeft
                    }
                }

                onReleased : file_dialog.open()
            }

            Button {
                id : _load
                highlighted : list_model.count
                font.capitalization : Font.Capitalize
                implicitWidth : _load_layout.implicitWidth + 20

                ToolTip.visible : hovered
                ToolTip.text : '????????? ????????? ?????? ?????? ??????'

                RowLayout {
                    id : _load_layout
                    anchors.fill : parent

                    Text {
                        text : '\ue890'
                        font.family : 'Material Icons Outlined'
                        font.pointSize : _load.font.pointSize + 4
                        Layout.alignment : Qt.AlignRight
                    }

                    Text {
                        text : 'Load'
                        font : _load.font
                        Layout.alignment : Qt.AlignLeft
                    }
                }

                onReleased : {
                    if (rb_reference.checked) {
                        con.read_reference_models();
                    } else if (rb_matrix.checked) {
                        con.read_matrices();
                    } else {
                        con.read_user_selected_model();
                    }
                }
            }
        }

        Rectangle {
            id : rectangle
            Layout.fillHeight : true
            Layout.fillWidth : true
            color : '#F2F2F2'

            ScrollView {
                anchors.fill : parent
                anchors.margins : 10
                clip : true

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

    function update_files_list(list) {
        list.forEach(x => list_model.append({'list_text': x}))
    }
}
