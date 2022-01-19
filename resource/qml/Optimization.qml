import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Window 2.15
import QtQuick.Layouts 1.15

import 'custom'

import Backend 1.0


Item {
    id : optimization

    property var points_count: 3;
    Component.onCompleted : set_points_count(points_count);

    function set_points_count(count) {
        points_count = count
        var total = count * measurements_count.value
        if (list_model.count === total) {
            return
        }

        list_model.clear();
        con.clear_temperature_measurement();

        for (var idx = 0; idx < total; idx++) {
            var pnt_idx = idx % count
            var row = idx + 1
            row = row < 10 ? '0' + row + '.' : row + '.'

            list_model.append({
                'index_text': row,
                'point_text': 'Point ' + (
                    pnt_idx + 1
                ),
                'day_text': '1',
                'time_text': '00:00:00',
                'time_enabled': (pnt_idx === 0)
            })
        }
    }

    function set_best_matching_model(model, psi) {
        _model.text = model
        _psi.text = psi
    }

    RowLayout {
        anchors.fill : parent
        spacing : 10

        CustomBox {
            title : 'Temperature Measurement'
            Layout.fillWidth : false
            Layout.preferredWidth : 450

            ColumnLayout {
                anchors.fill : parent

                RowLayout {
                    Label {
                        text : 'Number of Measurements'
                    }

                    SpinBox {
                        id : measurements_count
                        from : 1
                        to : 99
                        value : 1

                        editable : true
                        validator : IntValidator {}

                        ToolTip.visible : hovered
                        ToolTip.text : '창호 시공부위 온도 실측 횟수'

                        onValueChanged : set_points_count(points_count)
                    }
                }

                Rectangle {
                    Layout.fillWidth : true
                    Layout.fillHeight : true

                    ScrollView {
                        anchors.fill : parent
                        clip : true

                        ScrollBar.vertical.policy : ScrollBar.AsNeeded
                        ScrollBar.horizontal.policy : ScrollBar.AlwaysOff

                        ListView {
                            id : list_view
                            anchors.fill : parent

                            model : ListModel {
                                id : list_model
                            }

                            delegate : TemperatureInput {
                                width : list_view.width - 25
                                height : 45

                                index : index_text
                                point.text : point_text
                                day.text : day_text
                                time.text : time_text
                                day.enabled : time_enabled
                            }
                        }
                    }
                }
            }
        }

        CustomBox {
            title : 'Optimization'

            ColumnLayout {
                anchors.fill : parent

                ColumnLayout {
                    Layout.fillHeight : true
                    Layout.fillWidth : true

                    FigureCanvas {
                        objectName : 'optimization_plot'
                        dpi_ratio : Screen.devicePixelRatio

                        Layout.fillWidth : true
                        Layout.fillHeight : true
                    }

                }

                RowLayout {
                    Layout.fillWidth : true

                    Button {
                        text : 'Optimize'
                        font.capitalization : Font.Capitalize
                        highlighted : true
                        onReleased : con.optimize()
                    }

                    Rectangle {
                        width : 50
                    }

                    Label {
                        text : 'Best Model'
                        font.pointSize : 14
                    }
                    TextField {
                        id : _model
                        readOnly : true
                        selectByMouse : true
                        horizontalAlignment : TextField.AlignHCenter
                        font.pointSize : 14
                    }

                    Rectangle {
                        width : 50
                    }

                    Label {
                        text : 'Linear Thermal Transmittance (𝝍)'
                        font.pointSize : 14
                    }
                    TextField {
                        id : _psi
                        readOnly : true
                        selectByMouse : true
                        horizontalAlignment : TextField.AlignHCenter
                        font.pointSize : 14
                    }
                    Label {
                        text : 'W/mK'
                        font.pointSize : 14
                    }
                }
            }
        }
    }
}
