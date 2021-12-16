import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Window 2.15
import QtQuick.Layouts 1.15

import 'custom'

import Backend 1.0


Item {
    id : optimization

    property var points_count: 4
    Component.onCompleted : set_points_count(points_count)

    function set_points_count(count) {
        points_count = count
        var total = count * parseInt(measurements_count.value)
        if (list_model.count === total) {
            return
        }

        list_model.clear();
        con.clear_temperature_measurement();

        for (var idx = 0; idx < total; idx++) {
            var pnt_idx = idx % count
            list_model.append({
                'index_text': (idx + 1) + '.',
                'point_text': 'Point ' + (
                    pnt_idx + 1
                ),
                'day_text': '0',
                'time_text': '00:00:00',
                'time_enabled': (pnt_idx === 0)
            })
        }
    }

    // function set_time() {
    //     var day = ''
    //     var time = ''
    //     for (var idx = 0; idx < list_model.count; idx++) {
    //         if (idx % points_count == 0) {
    //             day = list_model.get(idx).day_text
    //             time = list_model.get(idx).time_text
    //         } else {
    //             list_model.set(idx, {
    //                 'day_text': day,
    //                 'time_text': time
    //             })
    //         }
    //         con.log('INFO|' + idx + '|' + day + '|' + time)
    //     }
    // }

    RowLayout {
        anchors.fill : parent
        spacing : 10

        ColumnLayout {
            Layout.fillHeight : true
            Layout.fillWidth : false
            Layout.preferredWidth : 500

            OptionItem {
                id : measurements_count

                Layout.fillHeight : false
                label.text : 'Number of Measurements'
                option_id : 'measurements count'
                value : '1'
                validator : IntValidator {}
                text_field.onEditingFinished : {
                    set_points_count(points_count)
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
                            // ListElement {
                            //     index_text : '0.'
                            //     point_text : 'Point 1'
                            //     day_text : '0'
                            //     time_text : '00:00:00'
                            //     time_enabled : true
                            // }
                        }

                        delegate : TemperatureInput {
                            width : list_view.width
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

            Button {
                text : 'Optimize'

                onReleased : con.optimize()
            }
        }

        ColumnLayout {
            Layout.fillHeight : true
            Layout.fillWidth : true

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

                Label {
                    text : '최적 모델'
                }
                TextField {
                    text : ''
                    readOnly : true
                }
                Label {
                    text : '선형 열관류율 (Ψ)'
                }
                TextField {
                    text : ''
                    readOnly : true
                }
                Label {
                    text : 'W/mK'
                }
            }
        }
    }
}
